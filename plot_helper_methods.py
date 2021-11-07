import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import math

labelFont = 18
legendFont = 16
tickFont = 16
latexDec=4

def plot_corr_matrix(mat, ticks=None, title='', ticks_fontsize=16, figsize=(10,6)):
    '''
    Plot a correlation matrix
    '''
    #%matplotlib inline
    sns.set(style="white")
    mask = np.zeros_like(mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    if ticks==None:
        ticks=mat.columns.values
    fix, ax =plt.subplots(figsize=figsize)
    sns.heatmap(mat, mask=mask, center=0, cmap=cmap,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=ticks, yticklabels=ticks)
    plt.tick_params(axis='both', which='major', labelsize= ticks_fontsize)
    plt.tick_params(axis='x', which='major', rotation=70)
    plt.title(title, fontsize=20)
    plt.show()


def plot_smd(random_smd, matched_smd):
    colors = list('rg')
    #labels = ['Before matching', 'After matching']
    plt.figure(figsize=(10,5))
    i = 1
    max_val = max(np.array(list(random_smd.values())).max(),
                     np.array(list(matched_smd.values())).max())
    for k in random_smd.keys():
        plt.hlines(y=i,xmin=0, xmax=max_val, 
                   color='lightgrey', alpha=.5)
        plt.plot( random_smd[k], i, marker = 'o', linewidth=3, markersize= 15, color='steelblue')
        plt.plot( matched_smd[k], i, marker = 'D',linewidth=3, markersize= 15, color='darkorange', alpha=0.7)
        i+=1
    plt.xticks(fontsize=tickFont)
    plt.yticks(range(1,1+len(matched_smd.keys())), matched_smd.keys(), fontsize=tickFont)
    
    oval = mlines.Line2D([], [], color='steelblue', marker='o', linestyle='None',
                          markersize=10, label='Random matching')
    diamond = mlines.Line2D([], [], color='darkorange', marker='D', linestyle='None', 
                          markersize=10, label='K-NN matching', alpha=0.7)


    plt.legend(handles=[oval,diamond], fontsize=legendFont)
    plt.xlabel('Mean difference', fontsize=labelFont)
    plt.ylabel('Covariates', fontsize=labelFont)
    plt.tight_layout()
    plt.show()
    

def plot_scree(n_comp, pca, forLatex = False, saveFile=None, figsize=(6,4)):

    fig = plt.figure(figsize=figsize)

    if(forLatex):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', serif='Times')
        fig.set_size_inches(get_fig_size(manuscriptColSize))

    plt.plot(range(1,1+n_comp), np.cumsum(pca.explained_variance_ratio_), 'o--', 
             color='orange', linewidth=2, label='cum. var. explained')
    plt.plot(range(1,1+n_comp), pca.explained_variance_ratio_, 'o-', linewidth=2 ,label='var. explained')
    plt.legend(fontsize=legendFont-forLatex*latexDec, markerscale=0.75, loc=0, frameon=True)
    plt.xlabel('Components',fontsize=labelFont-forLatex*latexDec)
    plt.ylabel('Variance explained',fontsize=labelFont-forLatex*latexDec)
    plt.xticks(range(1,1+n_comp), fontsize=tickFont-forLatex*latexDec)
    plt.yticks([0,.2,.4,.6,.8,1], fontsize=tickFont-forLatex*latexDec)

    plt.tight_layout()

    if (saveFile):
        plt.savefig(saveFile,dpi=100)
    else:
        plt.show()


def plot_factor_loadings(loadings, items, threshold=.3, 
    save_file=None, forLatex=False):
    markers = list("ovd^<>hp+x")
    colors = ['b', 'c', 'y', 'm', 'lightcoral', 'cornflowerblue','violet','teal']
    fig = plt.figure()
    if(forLatex):
        print('setting latex size')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', serif='Times')
        w, h =get_fig_size(manuscriptColSize)
        fig.set_size_inches((w,w))
    else:
        fig.set_size_inches((8,8))
        
#     ax = fig.add_subplot(111)
#     ax.yaxis.tick_right()
        
    i=0
    for v in items:
        if math.fabs(loadings.loc[0][v])>=threshold or math.fabs(loadings.loc[1][v])>=threshold:
            x=loadings.loc[0][v]
            y=loadings.loc[1][v]
            label = v+'\n({:.2f},{:.2f})'.format(x,y) if v=='Number of people' else v+' ({:.2f},{:.2f})'.format(x,y)
            plt.scatter(x+.01, y-.01,  
                        label= label, s=90, color='gray'
                        #, color=colors[i], marker=markers[i]
                        )
            plt.text(x, y, v, color='black')
            i+=1
    plt.axhline(y=0, linestyle='--', color='lightgray')
    plt.axvline(x=0, linestyle='--', color='lightgray')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    #plt.xticks(np.arange(-1,1.25,.25), fontsize=tickFont-forLatex*latexDec-2, rotation=90)
    #plt.yticks(np.arange(-1,1.25,.25), fontsize=tickFont-forLatex*latexDec-2)
    #plt.tick_params(labelsize= tickFont-forLatex*latexDec-2)
    plt.xlabel('Factor1',fontsize=labelFont-forLatex*latexDec-2)
    plt.ylabel('Factor2',fontsize=labelFont-forLatex*latexDec-2)
    # plt.legend(fontsize=legendFont-forLatex*latexDec-2, markerscale=0.4, loc=2, 
    #         handletextpad=.01, bbox_to_anchor=(-0.03, 1.02),
    #       ncol=1, frameon=True#,fancybox=True, shadow=True
    #           )
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file,dpi=300)
    plt.show()

