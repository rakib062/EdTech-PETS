import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from io import StringIO
import pandas as pd


def test_logit(data, predictors, label='label', normalize=True, l1_lambda = 0):
    X = data[predictors]
    if normalize:
        X = X.apply(stats.zscore)
    X = sm.add_constant(X)
    y = data[label]
    return sm.Logit(endog=y, exog=X).fit_regularized(
        disp = False, method='l1', alpha=l1_lambda, trim_mode='size', size_trim_tol=.1, maxiter=10000)

def get_logit_model(data, predictors, label='label', normalize=True, l1_lambda = 0):
    return test_logit(data, predictors, label=label, normalize=normalize, l1_lambda = l1_lambda)

def print_chisq(model):
    print('\nChisq:{:.2f}, p:{:.2f}\n'.format(model.llr, model.llr_pvalue))
    
def get_Rsq(model):
    return (model.llr) / (- 2*model.llnull)

def get_model_summary(model):
    summary = dict()
    summary['Chi^2'] = model.llr
    summary['p(Chi^2)'] = model.llr_pvalue
    summary['R^2'] = (model.llr) / (- 2*model.llnull)
    return summary

def get_OR(model):
#     output = model.conf_int()
#     output['OR'] = model.params
#     output.columns = ['2.5%', '97.5%', 'OR']
#     output = np.exp(output)[['OR', '2.5%', '97.5%']]
    
    '''add coef and p-values'''
    t=model.summary().tables[1]
    output = pd.read_csv(StringIO(t.as_csv()), sep=",")
    output.columns=[c.strip() for c in output.columns]
    output['OR'] = np.exp(output['coef'])
    output['_[0.025'] = np.exp(output['[0.025'])
    output['0.975]_']=np.exp(output['0.975]'])
    return output

def test_logit_model_accuracy(df, predictors, label, max_iter = 100, 
                              average=None, pos_label=1, penalty='none', c=1, class_weight=None):
    X_train, X_test, y_train, y_test = \
    train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)
    clf = LogisticRegression(random_state=0, max_iter=max_iter,
                            penalty=penalty, C=c, class_weight=class_weight).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = 100 * len(predictions[predictions==y_test])/len(y_test)
    precision, recall, fscore, support = score(y_test, predictions, average=average, pos_label=pos_label)

    c_m = confusion_matrix(y_test, predictions)
    
    print("\
        Accuracy:{:.2f}\n\
        Precision:{}\n\
        Recall:{}\n\
        F1:{}\n\
        support:{}".format( accuracy, precision, recall, fscore,support))
    
    print('Confusion matrix:')
    print(c_m)
    
    return clf