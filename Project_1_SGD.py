

# ======================================================START====================================================== #
import numpy as np
import scipy.stats as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics  import roc_curve,auc,roc_auc_score
from sklearn.metrics import precision_recall_curve



# First, load the data. Here we just use the train data and remove the column 1 and 28.

def loadData(fname):
    data = pd.read_csv(fname,header = None)
# Since we encode each column from 0, in order to delete columns 1 and 28, we should delete columns numbered 0 and 27.
    data = data.drop(27,axis=1)
    data = data.drop(0,axis=1)
    
    return data



# ======================================================Task 1====================================================== #

# Now, we use the Seaborn library to show the relationship between variables (correlation coefficient).


def plot_correlation(data):
    sns.pairplot(data) 
    plt.savefig('Relationship.png')
"""
The the diagonals above show the distributions of variables, from here we can know that some variables 
are continuous variables as well as the others are discrete variables, which means that we should 
consider this difference when we select the classification models.
"""



# ======================================================Task 2====================================================== #

"""
First of all, let's split the dataset into two parts: train set and test set. 
Secondly, in order to investigate the effect of Adam and SGD on training and test performance, 
we use these two methods to fit the model respectively, and then observe the value of accuracy, 
precision, recall and f1 score.
"""

def compute_SGD_Adam(data,lri =0.1,hiddens = [20],maxiter = 10000):
    
# First of all, since the last column is the dependent variable, so we set Y equals to it. 
    X = data.values[:,:-1] # colunm 1 to 26
    Y = data.values[:,-1] # colunm 28
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.4) # train set : test set = 6:4
 
 # Secondly, data standardization
 # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html  
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x) # Fit to data, then transform it.
    test_x = ss.transform(test_x) # Perform standardization by centering and scaling

 # Finally, compute the accuracy, precision, recall and f1 score of two methods  
 # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html  

 # For SGD: 
    mlp_SGD = MLPClassifier(hiddens,learning_rate_init= lri,activation='relu',solver='sgd', alpha=0.0001,max_iter = maxiter) 
    mlp_SGD.fit(train_x,train_y)
    ypred_sgd = mlp_SGD.predict(test_x)
    acc_sgd = metrics.accuracy_score(test_y, ypred_sgd)
    pres_sgd = metrics.precision_score(test_y, ypred_sgd, average='micro') 
    # "micro": Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html?highlight=metrics%20precision_score
    f1_sgd = metrics.f1_score(test_y, ypred_sgd)
    recall_sgd = metrics.recall_score(test_y, ypred_sgd)    

# For Adam:  
    mlp_Adam = MLPClassifier(hiddens,learning_rate_init= lri ,activation='relu',solver='adam', alpha=0.0001,max_iter=maxiter)  
    mlp_Adam.fit(train_x,train_y)
    ypred_adam = mlp_Adam.predict(test_x)
    acc_adam = metrics.accuracy_score(test_y, ypred_adam)
    pres_adam = metrics.precision_score(test_y, ypred_adam, average='micro') 
    # "micro": Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html?highlight=metrics%20precision_score
    f1_adam = metrics.f1_score(test_y, ypred_adam)
    recall_adam = metrics.recall_score(test_y, ypred_adam)
    
    
    return np.array([acc_sgd,pres_sgd,f1_sgd,recall_sgd]),np.array([acc_adam,pres_adam,f1_adam,recall_adam])


"""
Since each case should require 10 experimental runs, so here N=10, and also since we have 4 indicators 
(accuracy, precision, recall and f1 score), here we set (N,4)
"""

def ten_test_SGD_Adam(data,N=10):
    
    set_sgd = np.zeros((N,4))
    set_adam = np.zeros((N,4))
    for i in range(N):
        test_sgd,test_adam = compute_SGD_Adam(data)
        set_sgd[i,:] = test_sgd
        set_adam[i,:] = test_adam
     
# Compute the means of 4 indicators after 10 experimental runs       
    average_sgd = np.average(set_sgd,axis=0)
    average_adam = np.average(set_adam,axis=0)
    
# Compute the 95% confidence interval of 4 indicators after 10 experimental runs 
# https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers   
    interval_sgd = st.t.interval(0.95, len(set_sgd)-1, loc=np.mean(set_sgd), scale=st.sem(set_sgd))
    interval_adam = st.t.interval(0.95, len(set_adam)-1, loc=np.mean(set_adam), scale=st.sem(set_adam))
   
    
    return average_sgd,average_adam,interval_sgd,interval_adam


"""
After comparing the results of tow fit models, we find that SGD performs better than Adam.
Therefore, here we build the neural network with SGD method.
"""

def neural_network_SGD(data,N =10,lri =0.09,hiddens = [10,6],maxiter = 10000,momentum_rate = 0.9):
    
    X = data.values[:,:-1]
    Y = data.values[:,-1]
    
    set_sgd = np.zeros((N,4))
    
    for i in range(N):
        train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.4)
        # Data Standardization
        ss = StandardScaler()
        train_x = ss.fit_transform(train_x)
        test_x = ss.transform(test_x)

        # Adam fit
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html        
        mlp_SGD = MLPClassifier(hiddens,learning_rate_init= lri,activation='relu',solver='sgd', momentum=momentum_rate,alpha=0.0001,max_iter = maxiter) 
        mlp_SGD.fit(train_x,train_y)
        ypred_sgd = mlp_SGD.predict(test_x)
        acc_sgd = metrics.accuracy_score(test_y, ypred_sgd)
        pres_sgd = metrics.precision_score(test_y, ypred_sgd, average='micro')
        f1_sgd = metrics.f1_score(test_y, ypred_sgd)
        recall_sgd = metrics.recall_score(test_y, ypred_sgd)
    
        test = np.array([acc_sgd,pres_sgd,f1_sgd,recall_sgd])        
        set_sgd[i,:] = test
    # Since questions 2 to 5 all need compute the mean and the 95% confidence interval, here we have: 
    average_sgd = np.average(set_sgd,axis=0) # Calculated by row direction
    interval_sgd = st.t.interval(0.95, len(set_sgd)-1, loc=np.mean(set_sgd), scale=st.sem(set_sgd))
    
    return average_sgd,interval_sgd[0],interval_sgd[1] # Here 0 means the upper limit of 95% confidence interval and 1 means lower limit



# ======================================================Task 3====================================================== #

def LM_SGD(data):
	# Learning rate
    learning_rate = (np.arange(10)+1)*0.1  # Since in neural_network_Adam N=10.
    average_learning_rate = np.zeros((len(learning_rate),4))
    lower_learning_rate =  np.zeros((len(learning_rate),4))
    upper_learning_rate = np.zeros((len(learning_rate),4))
    
    
    for i in range(len(learning_rate)):
        test_learning_rate = learning_rate[i]
        average_learning_rate[i,:],lower_learning_rate[i,:],upper_learning_rate[i,:] = neural_network_SGD(data,lri = test_learning_rate)
    # Plot learning rate of 4 indicators    
    plotPlot(learning_rate,average_learning_rate,lower_learning_rate,upper_learning_rate,xlabel ='Learning Rate')
    plt.savefig('Learning Rate.png')

    
    # Momentum rate
    momentum_rate = (np.arange(10)+1)*0.01
    average_momentum_rate = np.zeros((len(momentum_rate),4))
    lower_momentum_rate =  np.zeros((len(momentum_rate),4))
    upper_momentum_rate = np.zeros((len(momentum_rate),4))
    
    
    for i in range(len(momentum_rate)):
        test_momentum_rate = momentum_rate[i]
        average_momentum_rate[i,:],lower_momentum_rate[i,:],upper_momentum_rate[i,:] = neural_network_SGD(data,momentum_rate = test_momentum_rate)
    # Plot momentum rate of 4 indicators    
    plotPlot(momentum_rate,average_momentum_rate,lower_momentum_rate,upper_momentum_rate,xlabel ='Momentum Rate')
    plt.savefig('Momentum Rate.png')

    return learning_rate,average_learning_rate,lower_learning_rate,upper_learning_rate,momentum_rate,average_momentum_rate,lower_momentum_rate,upper_momentum_rate

# The best learning rate is 0.1 and the best momentum rate is 0.08



# ======================================================Task 4====================================================== #
# Investigate the effect of different combinations of the number of hidden neurons (eg. 5, 10, 15, 20, 25) for single hidden layer. 

def number_of_hidden_neurons_SGD(data):
    neurons = (np.arange(5)+1)*5
    average = np.zeros((len(neurons),4))
    lower =  np.zeros((len(neurons),4))
    upper = np.zeros((len(neurons),4))   
    
    for i in range(len(neurons)):
        test_neurons = neurons[i]
        average[i,:],lower[i,:],upper[i,:] = neural_network_SGD(data,hiddens=test_neurons)
        
    plotPlot(neurons,average,lower,upper,xlabel ='The number of hidden neurons for single hidden layer')
    plt.savefig('The number of hidden neurons for single hidden layer.png')

    return neurons,average,lower,upper

# The bets number of hidden neurons is 20.

# ======================================================Task 5====================================================== #
"""
# Investigate the effect on a different number of hidden layers (1, 2, 3, 4) with a selected number of hidden neurons 
for any of the selected data sets.
"""

def number_of_hidden_layers_SGD(data):
    layers = [[25],[25,20],[25,20,15],[25,20,15,10]]
    average = np.zeros((len(layers),4))
    lower =  np.zeros((len(layers),4))
    upper = np.zeros((len(layers),4))   
    
    for i in range(len(layers)):
        test_layers = layers[i]
        average[i,:],lower[i,:],upper[i,:] = neural_network_SGD(data,hiddens=test_layers)
        
    plotPlot(np.arange(len(layers))+1,average,lower,upper,xlabel ='The number of hidden layers with a selected number of hidden neurons')
    plt.savefig('The number of hidden layers with a selected number of hidden neurons.png')

    return layers,average,lower,upper

# The best number of hidden layers is 3 which means [25,20,15]


# Since question 4 and 5 need to plot figures, therefore, here we create a function to plot which called plotPlot
# https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.plot.html

def plotPlot(x,y1,y2,y3,xlabel = '',title = ''):
    fig = plt.figure(figsize = (9,18))
    ylabels = ['Accuracy','Precision','F1','Recall']
    
    for i in range(y1.shape[1]):
        ax = fig.add_subplot(4,1,i+1)
        plt.plot(x,y1[:,i])
        plt.plot(x,y2[:,i],'r-.')
        plt.plot(x,y3[:,i],'r--')
        plt.ylabel(ylabels[i])  
    plt.xlabel(xlabel)
    plt.title(title)

    return 0
    


# ======================================================Task 6====================================================== #
"""
Evaluate the best model using a confusion matrix and show ROC and AUC, Precision-Recall curve and F1 Score 
for the classification problem.
Here, the learning rate is 0.1, the momentum rate is 0.08, the number of hidden layers is 3 which means [25,20,15].
"""

def neural_network_SGD_best(data,N =1,lri = 0.1,hiddens = [25,20,15],maxiter = 10000,momentum_rate = 0.08):
    
    X = data.values[:,:-1]
    Y = data.values[:,-1]
    set_sgd= np.zeros((N,4))
    
    for i in range(N):
        train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.4)
        
        ss = StandardScaler()
        train_x = ss.fit_transform(train_x)
        test_x = ss.transform(test_x)
             
        mlp_SGD = MLPClassifier(hiddens,learning_rate_init= lri,activation='relu',solver='sgd', momentum=momentum_rate,alpha=0.0001,max_iter = maxiter)       
        mlp_SGD.fit(train_x,train_y)
        ypred_sgd = mlp_SGD.predict(test_x)
        acc_sgd = metrics.accuracy_score(test_y, ypred_sgd)
        pres_sgd = metrics.precision_score(test_y, ypred_sgd, average='micro')
        f1_sgd = metrics.f1_score(test_y, ypred_sgd)
        recall_sgd = metrics.recall_score(test_y, ypred_sgd)
                                         
        # Plot confusion matrix.
        test = np.array([acc_sgd,pres_sgd,f1_sgd,recall_sgd])
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        confusion_matrix_test = confusion_matrix(test_y.flatten(), ypred_sgd.flatten(), labels=[0,1]) # List of labels to index the matrix. 
        sns.set()
        set_sgd[i,:] = test
        
        f,ax=plt.subplots()
        sns.heatmap(confusion_matrix_test,annot=True,ax=ax)
        ax.set_title('confusion matrix') 
        ax.set_xlabel('predict') 
        ax.set_ylabel('true') 
        plt.savefig('confusion matrix.png')

        
        # Print Precision-Recall curve and F1 Score, ROC and AUC
        print(classification_report(ypred_sgd,test_y,target_names = ['0','1']))
        
        # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, ypred_sgd)
        print(auc(false_positive_rate, true_positive_rate)) # roc_auc_score
        
        roc_auc = auc(false_positive_rate,true_positive_rate) # Compute the AUC
         
        plt.figure()
        plt.figure(figsize = (10,10))
        plt.plot(false_positive_rate, true_positive_rate, color = 'deeppink',
                 lw = 2, label='ROC curve (area = %.2f)' % roc_auc) # X: false positive rate, Y: true positive rate
        plt.plot([0, 1], [0, 1], color = 'darkgreen', lw = 2, linestyle = '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('plot_AUC')
        plt.legend(loc = "lower right")
        plt.savefig('plot_AUC.png')

        # Plot the Precision-Recall curve
        # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
        plt.figure()
        pr_precision, pr_recall, _ = precision_recall_curve(test_y, ypred_sgd)
        # summarize scores
        print('Logistic: f1=%.3f' % (f1_sgd))
        # plot the precision-recall curves
        no_skill = len(test_y[test_y==1]) / len(test_y)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(pr_recall, pr_precision, marker='.', label='Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.savefig('Precision-Recall curve.png')
        
    return set_sgd
  


# ======================================================ANSWERS====================================================== #

if __name__ =='__main__':
    
    
# load the data
    data = loadData('train_data.txt')

    
# Task 1
    plot_correlation(data) 
    
    
# Task 2 
    average_sgd,average_adam,interval_sgd,interval_adam = ten_test_SGD_Adam(data,N=10)
    print(average_sgd)
    print(average_adam)
    print(interval_sgd)
    print(interval_adam)

   
# Task 3  
    LM_SGD(data)

   
# Task 4
    number_of_hidden_neurons_SGD(data)
    

# Task 5 
    number_of_hidden_layers_SGD(data)
    
    
# Task 6 
    neural_network_SGD_best(data)
    


# =========================================================END========================================================= #    
    
    
    
