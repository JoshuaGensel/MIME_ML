from simDataPreprocessing import convertToMatrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def testRegMod(path : str, nSeq : int) -> plt:
    """
    This function trains a logistic regression model on the dmMIMESim data and plots the weights of the model vs the log of the kd values for each residue.
    
    First the simualted sequences get loaded using the convertToMatrix function. Then the sequences are split into training and test sets. The training set 
    is used to train the logistic regression model. The test set is used to evaluate the model. The model is trained using a grid search to find the best 
    parameters. The best parameters are then used to train the model. The weights of the model are then plotted against the log of the kd values for each 
    residue. The function returns the plot.

    Args:
        path (str): string of path to a folder containing dmMIMESim output files for protein concentrations 1, 6, 15 and 30 called /prot1, /prot6, /prot15, /prot30.
        maxNumSequences (int): the maximum number of sequences to use for training the model. If the number of sequences is less than this number, the model will be trained on all sequences.

    Returns:
        plt: plot of the weights of the model vs the log of the kd values for each residue.
    """
    
    #load sequence data as numpy array with convertToMatrix function for every protein concentration
    prot1_bound = convertToMatrix(path+'/prot1/sequences/3.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=1, seed=12678435)
    prot1_unbound = convertToMatrix(path+'/prot1/sequences/4.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=1, seed=25681)

    prot6_bound = convertToMatrix(path+'/prot6/sequences/3.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=6, seed=71092485)
    prot6_unbound = convertToMatrix(path+'/prot6/sequences/4.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=6, seed=481276)
    
    prot15_bound = convertToMatrix(path+'/prot15/sequences/3.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=15, seed=2671)
    prot15_unbound = convertToMatrix(path+'/prot15/sequences/4.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=15, seed=92178)
    
    prot30_bound = convertToMatrix(path+'/prot30/sequences/3.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=30, seed=28154)
    prot30_unbound = convertToMatrix(path+'/prot30/sequences/4.txt', maxNumSequences=nSeq,splitProb=3/450, readSize=100, protConc=30, seed=1823467)
    
    #combine all bound and unbound data into one array
    bound = np.concatenate((prot1_bound, prot6_bound, prot15_bound, prot30_bound), axis=0)
    unbound = np.concatenate((prot1_unbound, prot6_unbound, prot15_unbound, prot30_unbound), axis=0)

    #create labels for bound and unbound data
    bound_labels = np.ones((bound.shape[0],1))
    unbound_labels = np.zeros((unbound.shape[0],1))

    #combine bound and unbound data and labels into one array
    data = np.concatenate((bound, unbound), axis=0)
    labels = np.concatenate((bound_labels, unbound_labels), axis=0)
    
    #create training and test data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)
    
    #train logistic regression model with elastic net regularization
    param_grid = {'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]}
    grid = GridSearchCV(LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000), param_grid, cv=5)
    grid.fit(X_train, y_train.ravel())
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Test set score: {:.2f}".format(grid.score(X_test, y_test.ravel())))
    
    # read in kd values
    kds = np.loadtxt(path+'/prot1/single_kds.txt')
    #get model coefficients
    weights = grid.best_estimator_.coef_[0][1:]*-1
    #get correlation between log(kd) and model coefficients
    corr, _ = pearsonr(np.log(kds), weights)

    #plot correlation between log(kd) and model coefficients
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.scatterplot(x=np.log(kds), y=weights, linewidth=0.5)
    plt.ylabel('coefficient')
    plt.xlabel('log(kd)')
    plt.title('correlation: '+str(round(corr,3)))
    plt.show()

    return plt