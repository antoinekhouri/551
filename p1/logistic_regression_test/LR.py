import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

#import data
df1 = pd.read_csv('data/ionosphere.data', header=None)

#Data Cleaning
#1. Eliminate duplicates
df1= df1.drop_duplicates(subset=None, keep='first', inplace=False)
#2.Check categorical vars for consistency
print(df1[34].value_counts())
#3. One hot encoding
dummy= pd.get_dummies(df1[34])
dummy.head()
df1=pd.concat([df1,dummy], axis=1)
df1.head()
#Drop non-encoded columns
df1=df1.drop(columns=[34])
#4.Eliminate missing values
df1=df1.dropna()

#Numpy objects
dataset1=df1.values

#Logistic Regression
class log_reg:
    def init(self,learning_rate,iterations, threshold):
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.threshold=threshold
    
    #Define the logistic function->binds y btw 0 and 1
    def logistic(self,z):
        sigma=1/(1+np.exp(-z))
        return sigma
    
    #Define the cost function J 
    def cost(self, yh, y):
        #z = np.dot(X,w) 
        J = np.mean( y * np.log1p(np.exp(yh)) + (1-y) * np.log1p(np.exp(yh)) )
        return J
    
    #Add the bias term (y-int) to the beginning of the X (feature) matrix.
    def add_bias(self, X):
        bias = np.ones((X.shape[0], 1))
        X_adjusted=np.concatenate((bias, X), axis=1)
        return X_adjusted
    
    #Define fit function
    def fit(self, X,y):
        #Add the bias term if necessary/wanted-> This is defined when calling the class
        X=self.add_bias(X)
        #Initialize the weights vector     
        self.weights=np.zeros(X.shape[1])
        
        for i in range (self.iterations):
            #Define the logit
            z = np.dot(X, self.weights)
            #Define y-hat-> the predicted y value
            yh=self.logistic(z)
            loss = self.cost(yh, y)
            # Gradient descent- minimize the cost function by adjusting the weights
            grad_desc = np.dot(X.T, (yh - y)) / y.size
            #self.weights-= self.learning_rate * grad_desc
            self.weights= self.weights-(self.learning_rate * grad_desc)
            #Update parameters
            z = np.dot(X, self.weights)
            yh = self.logistic(z)
            loss_new = self.cost(yh, y)
            if abs(loss-loss_new) <= self.threshold:
                return self.weights, yh         
        return self.weights, yh
    def predict (self, X): 
        X = self.add_bias(X)
        yh=self.logistic(np.dot(X, self.weights))
        yh_rounded=np.zeros(yh.shape)
        for i in range(len(yh)):
            if yh[i]>=0.5:
                yh_rounded[i]=1
            else:
                yh_rounded[i]=0
        return yh_rounded
        
    def evaluate_acc(self, y,yh):
    # positive =1, negative=0
        TP,TN,FP,FN=0,0,0,0
        for i in range(len(y)):
            if y[i]==yh[i]==1:
                TP+=1
            if y[i]==yh[i]==0:
                TN+=1
            if y[i]!=yh[i]==1:
                FP+=1
            if y[i]!=yh[i]==0:
                FN+=1
    
        accuracy= (TP+TN)/(TP+FP+TN+FN)
        error= (FP+FN)/(TP+FP+TN+FN)
        return accuracy, error

#Cross validation Script
# folds=5
# DS_shuffle=dataset1
# np.random.shuffle(DS_shuffle)
# fold_length=round((np.shape(DS_shuffle)[0])/folds)
# st=int(0)
# acc=np.zeros(folds)
# err=np.zeros(folds)            
# for i in range(folds):
#     if i == folds:
#         end=np.shape(DS_shuffle)[0]
#     else:
#         end=st+fold_length
#     test=DS_shuffle[st:end,0:]
#     train=np.delete(DS_shuffle,slice(st,end),0)
#     #Fit model to training data
#     weights, yh= modelDS1.fit(train[0:,0:33],train[0:,34])
#     #Test data
#     yh_rounded= modelDS1.predict(test[0:,0:33])
#     acc[i], err[i]=modelDS1.evaluate_acc(test[0:,34], yh_rounded)                    
#     st+=fold_length