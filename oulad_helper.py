import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn import utils
from torch.autograd import Function
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.utils.data as data_utils
from gradrev import GradientReversal
# from audtorch.metrics.functional import pearsonr as pycorr
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support as performance_score
from sklearn.metrics import confusion_matrix


import random 

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

data_root='root-directory/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activity_name_dict = {'resource':'pdf_resource', #e.g., books
                      'oucontent':'assignment', 'url':'audio_video_link',
                      'homepage':'course_homepage','page':'course_info_page',
                      'subpage':'external_website', 'glossary':'glossary',
                      'forumng':'forumng',               
                      'oucollaborate':'online_video_discussion',
                      'dataplus':'external_resources',
                      'dualpane':'site_info',
                      'ouelluminate':'online_tutorial_session','quiz':'quiz',
                      'sharedsubpage':'shared_pages', #was not found significant predictor in prev paper
                      'questionnaire':'course_questionnaire',
                      'externalquiz':'external_quiz',
                      'ouwiki':'wiki_content',
                      'repeatactivity':'content_prev_weeks',
                      'folder':'course_files',
                      'htmlactivity':'interactive_html'}
                      
weekly_activity_dict = {'resource':'pdf_resource', #e.g., books
                      'url':'audio_video_link',
                      'page':'course_info_page',
                      'forumng':'forumng',
                      'dataplus':'external_resources',
                      'ouwiki':'wiki_content',
                      'repeatactivity':'content_prev_weeks',
                      'folder':'course_files',
                      'htmlactivity':'interactive_html'}                      


def create_data_loader(data, predictors, outcome_col, batch_size=10, validation=0.2):
    '''
    Prepare data by converting pandas to tensors and data loaders.
    
    Parameter:
        data: pandas dataframe
        predictors: list of predictor variables
        outcome_col: dataframe column containing outcome
        batchsize: size of batches
        validation: percentage of data that will be reserved for validation
    '''
    train_split, test_split = train_test_split(data, test_size=validation)
    
    train_tensor = data_utils.TensorDataset(torch.Tensor(train_split[predictors].values), 
                                            torch.Tensor(train_split[outcome_col].values))
    train_data_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    
    test_tensor = data_utils.TensorDataset(torch.Tensor(test_split[predictors].values), 
                                            torch.Tensor(test_split[outcome_col].values))
    test_data_loader = data_utils.DataLoader(test_tensor, batch_size=batch_size, shuffle=True)
    
    return train_data_loader, test_data_loader


class Predictor(nn.Module):
    '''
        A class containing Feature extractor for the 
        Open University Learning Analytics dataset.
    '''
    def __init__(self, dim_in):
        super(Predictor, self).__init__()

        '''The feature extractor'''
        self.feature_extractor = \
           nn.Sequential(
                nn.Linear(in_features = dim_in, out_features = 30, bias=True),
                nn.ReLU(),
                nn.Linear(in_features = 30, out_features = 20, bias=True),
                nn.ReLU(),
                nn.Linear(in_features = 20, out_features = 10, bias=True),
                nn.ReLU(),
                nn.Linear(in_features = 10, out_features = 10, bias=True),
                nn.ReLU(),
                nn.Linear(in_features = 10, out_features = 1, bias=True)
            )

    def forward(self, x):
        #print('forward called')
        features = self.feature_extractor(x)
        return features

    
def train_outcome_predictor(data, predictors, outcome_col, batch_size = 10,
                            n_epochs=15, validation=.2, class_weights=torch.FloatTensor([1])):
    '''
    Function to train a model to predict the outcome.
    Parameters
        data: pandas dataframe
        predictors: names of the predictor variables
        outcome_col: name of the outcome variable
        batch_size: batch size
        n_epochs: number of epochs
    '''

    train_data_loader, val_data_loader= create_data_loader(data, predictors, outcome_col, 
                                                           batch_size=batch_size, validation=validation)

    #create the model
    model = Predictor(dim_in = len(predictors)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights)
    
    class_accuracy = [] #classification accuracy per epoch for validation set
    val_size = len(data)*validation
    train_size = len(data)*(1-validation)
    for epoch in range(1, n_epochs+1):
        train_loss = 0

        #train
        for data, labels in tqdm(train_data_loader, leave=False):
            data = data.to(device)
            labels = labels.to(device)

            predictions = model(data).squeeze().view(-1)
            loss = loss_fun(predictions, labels)

            #remove gradients from previous pass
            optim.zero_grad()
            # compute accumulated gradients
            loss.backward()
            #update parameters based on current gradients
            optim.step()
            train_loss += loss.item() * data.size(0)

            
        #validate
        valid_loss = 0
        valid_accuracy=0
        for data, labels in tqdm(val_data_loader, leave=False):
            data = data.to(device)
            labels = labels.to(device)

            predictions = model(data).squeeze().view(-1)
            loss = F.binary_cross_entropy_with_logits(predictions, labels)
            valid_loss += loss.item() * data.size(0)
            
            class_output = (torch.sigmoid(predictions)>0.5).float()
            valid_accuracy += (class_output == labels).float().sum().item()
        
        class_accuracy.append(valid_accuracy/val_size)
        tqdm.write('EPOCH {}: train-loss:{:.4f}, val-loss:{:.4f}, val-accuracy= {:.4f}'.format(
                epoch, train_loss/train_size, valid_loss/val_size, valid_accuracy/val_size))

    return class_accuracy



class FeatExtractorGRLT(nn.Module):
    '''
        NN model with a Gradient Reversal Layer for domain adaptation. 
        It has a linear transformation layer from which transformed features can be extracted.
    '''
    def __init__(self, dim_in, lt_dim=10, lambda_=.3):
        
        '''
        Parameters:
            dim_in: input/feature dimension
            lt_dim: # of nodes of the linear transform layer
            lambda_: weight of gradient reversal
        '''
        
        super(FeatExtractorGRLT, self).__init__()
        
        '''The linear transform layer'''
        self.lt = nn.Linear(in_features = dim_in, out_features = lt_dim, bias=True)
        
        '''The encoder'''
        self.feature_extractor = \
            nn.Sequential(
#                 nn.BatchNorm1d(lt_dim),
                nn.ReLU(),
                nn.Linear(in_features = lt_dim, out_features = 20, bias=True),
#                 nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(in_features = 20, out_features = 10, bias=True),
#                 nn.BatchNorm1d(10),
                nn.ReLU()
            )

        '''predictor of the outcome variable'''
        self.outcome_predictor = nn.Sequential(
            nn.Linear(in_features = 10, out_features = 10, bias=True),
#             nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features = 10, out_features = 1, bias=True))
        
        '''Predictor of sensitive attribute (gender)'''
        print('lambda_ :', lambda_)
        self.sensitive_attr_predictor = nn.Sequential(
            GradientReversal(lambda_=lambda_), 
            nn.Linear(in_features = 10, out_features = 10, bias=True),
#             nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features = 10, out_features = 1, bias=True))
        

    def forward(self, x):
        transformed = self.lt(x)
        features = self.feature_extractor(transformed)
        return self.outcome_predictor(features), self.sensitive_attr_predictor(features)
    

class FeatExtractorGR(nn.Module):
    '''
        NN model with a Gradient Reversal Layer for domain adaptation. 
        It does not have a linear transformation layer from which transformed features can be extracted.
    '''
    def __init__(self, dim_in, lambda_=.3):
        
        '''
        Parameters:
            dim_in: input/feature dimension
            lambda_: weight of gradient reversal
        '''
        
        super(FeatExtractorGR, self).__init__()
        
        '''The linear transform layer'''
#         self.lt = nn.Linear(in_features = dim_in, out_features = lt_dim, bias=True)
        
        '''The encoder'''
        self.feature_extractor = \
            nn.Sequential(
#                 nn.BatchNorm1d(dim_in),
                nn.ReLU(),
                nn.Linear(in_features = dim_in, out_features = 20, bias=True),
#                 nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(in_features = 20, out_features = 10, bias=True),
#                 nn.BatchNorm1d(10),
                nn.ReLU()
            )

        '''predictor of the outcome variable'''
        self.outcome_predictor = nn.Sequential(
            nn.Linear(in_features = 10, out_features = 10, bias=True),
#             nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features = 10, out_features = 1, bias=True))
        
        '''Predictor of sensitive attribute (gender)'''
        print('lambda_ :', lambda_)
        self.sensitive_attr_predictor = nn.Sequential(
            GradientReversal(lambda_=lambda_), 
            nn.Linear(in_features = 10, out_features = 10, bias=True),
#             nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features = 10, out_features = 1, bias=True))
        

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.outcome_predictor(features), self.sensitive_attr_predictor(features)
    
def train_feat_transformer(data, predictors, outcome_col, 
                           sensitive_attr_col, model, n_epochs=15, class_weights=torch.FloatTensor([1]),
                           l1_weight=0., l2_weight=.0, ortho_weight=0., 
                           lasso=True, l1_mul=1, l1_ann=True):
    '''
    Train a model using domain adaptation (gradient reversal) technique. 
    It has a Linear Transformation layer right after the input layer from which intermediate features 
    will be extracted.
    
    Parameters
        data: pandas dataframe
        predictors: names of the predictor variables
        outcome_col: name of the outcome variable
        sensitive_attr_col: name of the column whose prediction accuracy will be minimized
        n_epochs: number of epochs
        l1_weight: weights for l1 shrinkage to get sparse weights 
        ortho_weight: weights for orthogonality constraints
        lasso: if True, apply lasso regression in the LT layer, otherwise apply constraints on the weights
        l1_mul: multiplicative factor for l1 weight.
    '''
    
    print('training model with l1={}, l2={}, ortho={}'.format(l1_weight, l2_weight, ortho_weight))
    
    batch_size = 10
    n_batches = math.ceil(len(data)/batch_size)
    sensitive_labels = torch.Tensor(data[sensitive_attr_col].values)
    outcome_labels = torch.Tensor(data[outcome_col].values)
    #create data loader    
    data_tensor = data_utils.TensorDataset(torch.Tensor(data[predictors].values), 
                                outcome_labels, sensitive_labels)
    data_loader = data_utils.DataLoader(data_tensor, 
                    batch_size=batch_size, shuffle=True)

    #create the an optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    domain_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights)
    
    domain_accuracy = [] #domain losses per epoch
    class_accuracy = [] #classification accuracy per epoch
    domain_losses = []
    class_losses= []
    d_precision, d_recall, d_f1, d_support = 0, 0, 0, 0
    for epoch in range(1, n_epochs+1):
        total_domain_loss = 0 
        total_label_loss = 0
        total_label_accuracy = 0
        total_domain_accuracy=0
        total_l1_loss=0
        #for (predictor, outcome, gender) triples
        for (x, y, s) in tqdm(data_loader, leave=False):
            x = x.to(device)
            domain_y = s.to(device)
            label_y = y.to(device)

            label_preds, domain_preds = model(x)
            label_preds = label_preds.squeeze().view(-1)
            domain_preds = domain_preds.squeeze().view(-1)
            
            domain_loss = domain_loss_fn(domain_preds, domain_y)
            label_loss = F.binary_cross_entropy_with_logits(label_preds, label_y)
            
            '''add L1 regularization loss for the linear transformation layer'''
            l1_norm = 0
            if l1_weight!=0:
                if lasso:
                    l1_norm = sum(param.abs().sum() for name,param in model.lt.named_parameters() \
                                  if 'weight' in name)
                else:
                    lt_params = list(model.lt.parameters())[0]
                    l1_norm = torch.mean(torch.abs(lt_params**2-lt_params)) #contraints = {0,1}
#             l1_norm = torch.mean(torch.abs(1 - torch.abs(lt_params)**torch.abs(lt_params*100))) #constraints={-1,0,1}
        
#             print(l1_norm.item(), domain_loss.item(), label_loss.item())
                
            '''add L2 regularization for other layers'''
            l2_norm = 0
            if l2_weight!=0:
                l2_norm += sum(param.pow(2.0).sum() for name, param in model.feature_extractor.named_parameters() \
                              if 'weight' in name)
                l2_norm += sum(param.pow(2.0).sum() for name,param in model.outcome_predictor.named_parameters() \
                              if 'weight' in name)
                l2_norm += sum(param.pow(2.0).sum() for name,param in \
                               model.sensitive_attr_predictor.named_parameters() if 'weight' in name)
                
            '''compute correlations among transformed features'''
            dotprod = 0
            if ortho_weight>0.:
                params=list(model.lt.parameters())[0]
                dotprod=torch.tensordot(params, params, dims=([1],[1])).abs().fill_diagonal_(0).sum()/2

            total_l1_loss += l1_weight*l1_norm 
            loss = domain_loss + label_loss + l1_weight*l1_norm + l2_weight*l2_norm + dotprod * ortho_weight
            

            #remove gradients from previous passes
            optim.zero_grad()
            # compute accumulated gradients
            loss.backward()
            #update parameters based on current gradients
            optim.step()

            total_domain_loss += domain_loss.item()
            total_label_loss += label_loss.item()
            
            class_output = (torch.sigmoid(label_preds)>0.5).float()
            total_label_accuracy += (class_output == label_y).float().sum().item()
            
            domain_output = (torch.sigmoid(domain_preds)>0.5).float()
            total_domain_accuracy += (domain_output == domain_y).float().sum().item()
            
        mean_domain_loss = total_domain_loss / len(data)
        mean_label_loss = total_label_loss / len(data)
        mean_class_accuracy = total_label_accuracy / len(data)
        mean_domain_accuracy = total_domain_accuracy / len(data)
        domain_accuracy.append(mean_domain_accuracy)
        class_accuracy.append(mean_class_accuracy)
        domain_losses.append(mean_domain_loss)
        class_losses.append(mean_label_loss)
        
        if l1_ann: # if annealin scheme is true for l1
            #l1_weight= l1_mul**epoch 
            l1_weight=1+epoch 
        
        tqdm.write('EPOCH {}: domain loss={:.4f}, class accuracy= {:.4f}, domain accuracy={:.4f}, l1 loss:{:.3f}, l1_weight:{}'.format(epoch, mean_domain_loss, mean_class_accuracy, mean_domain_accuracy, total_l1_loss, l1_weight))
#         tqdm.write("Domain\n\
#         Precision:{}\n\
#         Recall:{}\n\
#         F1:{}\n\
#         support:{}".format( d_precision/n_epochs, d_recall/n_epochs, d_f1/n_epochs, d_support/n_epochs))
    return (domain_accuracy, class_accuracy, domain_losses, class_losses)



def train_model(model=None, lt_dim = 10, l1_weight = 0.1, l1_mul=1, l2_weight = 0.1, l1_ann=True, lambda_=.3,
                ortho_weight=0, lasso=True, mask_lt=False, class_weights=torch.FloatTensor([1]),
                n_epochs=10, df=None, predictors=None, outcome=None, sensitive_attr=None, show_plots=True):

                
    if df is None:
        df = pd.read_csv(join(data_root,'semester-level-activity.csv'))
        df.set_index('row_id', inplace=True)
        predictors = list(set(df.columns).difference(set([
                        'gender_num', 'final_result_num', 'credits', 'highest_education', 
                        'previously_attempted', 'blocks' , 'final_result', 'gender', 'age_num'])))
        outcome = 'final_result_num'
        sensitive_attr = 'gender_num'
    
    trans_columns=['f_{}'.format(i) for i in range(1,1+lt_dim)]
    
    if model==None:
        model = FeatExtractorGRLT(dim_in=len(predictors), lt_dim=lt_dim, lambda_=lambda_).to(device)
#     params=list(model.lt.parameters())[0]
#     dotprod=torch.tensordot(params, params, dims=([1],[1])).abs().fill_diagonal_(0)#.sum()/2
#     print('initial dot product: ', dotprod)
#     print(dotprod.sum()/2)
    domain_acc, class_acc, domain_loss, class_loss = train_feat_transformer(
                                        data=df, 
                                        predictors=predictors,
                                        outcome_col=outcome,
                                        sensitive_attr_col=sensitive_attr,
                                        model=model,
                                        n_epochs=n_epochs,
                                        l1_weight = l1_weight,
                                        l2_weight = l2_weight,
                                        ortho_weight = ortho_weight,
                                        lasso=lasso, 
                                        l1_mul=l1_mul,
                                        l1_ann= l1_ann,
                                        class_weights=class_weights,
                                        )


    '''plot spreads of parameters'''
    ncol=5
    nrow=int(np.ceil(lt_dim/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 8))
    i=0
    j=0
    for in_params in list(model.lt.parameters())[0]:
        params = in_params.tolist()
        sns.boxplot(x=params, color='orange', ax=axes[i][j])
        sns.swarmplot(x=params, color='black',  size=3, ax=axes[i][j])
        axes[i][j].set_xlim([-1, 1])
        j+=1
        if j==ncol:
            j=0
            i+=1
    plt.title('l1: {}'.format(l1_weight))
    if show_plots:
        plt.show()
    else:
        plt.savefig('output/node_spread_l1_{:.2f}_l2:{:.2f}.pdf'.format(l1_weight, l2_weight))

    plt.figure(figsize=(20, 6))
    params = np.array(list(model.lt.parameters())[0].tolist())
    plt.boxplot(x=params, positions=range(1, len(predictors)+1), showmeans=True)
    plt.xticks(range(1, len(predictors)+1), rotation=90)
    plt.axhline(y=0, color='grey', alpha=.4)
    plt.title('l1:{}'.format(l1_weight))
    if show_plots:
        plt.show()
    else:
        plt.savefig('output/params_spread_l1_{:.2f}_l2_{:.2f}.pdf'.format(
                                                            l1_weight, l2_weight))

    '''now extract linearly transformed features from the model that was trained above'''

    transformd_feats = model.lt(torch.Tensor(df[predictors].values))
    transformed_feat_df =  pd.DataFrame(transformd_feats.detach().numpy(), 
                                    columns=trans_columns,
                                    index =df.index)

    if show_plots==False:
        transformed_feat_df.to_csv('transformed_feat_df_l1_{:.2f}_l2_{:.2f}_.csv'.format(
                                        l1_weight, l2_weight))

    '''plot domain and label accuracy'''
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    axes[0].plot(domain_acc, 'o-', label='domain acc. l1:{:.2f}'.format(l1_weight))
    axes[0].plot(class_acc, 'd--', label='class acc. l1:{:.2f}'.format(l1_weight))

    axes[1].plot(domain_loss, 'o-', label='domain loss l1:{:.2f}'.format(l1_weight))
    axes[1].plot(class_loss, 'd--', label='class loss l1:{:.2f}'.format(l1_weight))

    axes[0].set_xticks(range(n_epochs), minor=False)
    axes[0].set_xlabel('epoch')
    axes[0].legend()

    axes[1].set_xticks(range(n_epochs))
    axes[1].set_xlabel('epoch')
    axes[1].legend()
    if show_plots:
        plt.show()
    else:
        plt.savefig('output/accuracy_loss_l2_{:.2f}.pdf'.format(l2_weight))


    return model


        
class FeatExtractorGR_Mask(nn.Module):
    '''
        NN model with a Gradient Reversal Layer. 
        It has a linear transformation layer from which transformed features can be extracted.
    '''
    def __init__(self, dim_in, lt_dim=10, lambda_=.3):
        
        '''
        Parameters:
            dim_in: input/feature dimension
            lt_dim: # of nodes of the linear transform layer
            lambda_: weight of gradient reversal
            lt_mask: weight mask for the LT layer. if None, all entries are 1s. Note, dim_in must be equal to lt_dim
        '''
        
        super(FeatExtractorGR_Mask, self).__init__()
        
        
        '''The linear transform layer'''
        self.lt = nn.Linear(in_features = dim_in, out_features = lt_dim, bias=True)
        #self.lt.weight = torch.nn.parameter.Parameter((torch.ones(dim_in, lt_dim)))
        torch.nn.init.normal_(self.lt.weight, mean=0.5, std=0.5)
        
        '''The encoder'''
        self.feature_extractor = \
            nn.Sequential(
#                 nn.BatchNorm1d(lt_dim),
                nn.ReLU(),
                nn.Linear(in_features = lt_dim, out_features = 20, bias=True),
#                 nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(in_features = 20, out_features = 10, bias=True),
#                 nn.BatchNorm1d(10),
                nn.ReLU()
            )

        '''predictor of the outcome variable'''
        self.outcome_predictor = nn.Sequential(
            nn.Linear(in_features = 10, out_features = 10, bias=True),
#             nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features = 10, out_features = 1, bias=True))
        
        '''Predictor of sensitive attribute (gender)'''
        self.sensitive_attr_predictor = nn.Sequential(
            GradientReversal(lambda_=lambda_), 
            nn.Linear(in_features = 10, out_features = 10, bias=True),
#             nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features = 10, out_features = 1, bias=True))
        

    def forward(self, x):
        weight = np.array(self.lt.weight.data)
        self.lt.weight = torch.nn.parameter.Parameter((torch.tensor(np.diag(np.diag(weight)))))
            
        transformed = self.lt(x)
        features = self.feature_extractor(transformed)
        return self.outcome_predictor(features), self.sensitive_attr_predictor(features)