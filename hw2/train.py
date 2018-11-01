import numpy as np
import random
import csv
import math

train_X = np.genfromtxt('train_x.csv', delimiter=',')
train_X = train_X[1:,:]
train_Y = np.genfromtxt('train_y.csv', delimiter=',')
train_Y = train_Y[1:]

################# input data cleaning ################
def cleandata(x):
    datasize = x.shape[0]
      
    # delete feature: SEX, MARRIAGE, BILL_AMT 2-6, PAY_AMT 4-6
    d = [1, 3, 12, 13, 14, 15, 16, 20, 21, 22]
    x = np.delete(x, d, 1)
    
    # one-hot encoding:  EDUCATION
    left = int(np.min(x[:,1]))
    right = int(np.max(x[:,1])+1)
    feature = x[:,1].copy()
    x = np.delete(x, 1, 1)
    for i in range(left, right):
        new = np.zeros((datasize,1))
        new[np.where(feature == i)] = 1
        x = np.hstack((x,new))

    x[x<0] = 0
    
    # normalize LIMIT_BAL, BILL_AMT and PAY_AMT
    norm = [0]+list(range(8,12))
    for i in range(len(norm)):
        Min = np.min(x[:,norm[i]])
        div = np.max(x[:,norm[i]])-Min
        x[:,norm[i]] -= Min
        x[:,norm[i]] /= div
 
    return x
    
train_X = cleandata(train_X)
datasize, datalength = train_X.shape

########### Generative Model : training ###########
train_X_c0 = train_X[np.where(train_Y==0)]
train_X_c1 = train_X[np.where(train_Y==1)]
numData_c0 = train_X_c0.shape[0]
numData_c1 = train_X_c1.shape[0]
average_c0 = np.average(train_X_c0, axis=0)
average_c1 = np.average(train_X_c1, axis=0)
covmat_c0 = np.cov(train_X_c0.T)
covmat_c1 = np.cov(train_X_c1.T)
covmat = (covmat_c0*numData_c0 + covmat_c1*numData_c1)/(numData_c0 + numData_c1)
covmat_inv = np.linalg.inv(covmat)

w = np.matmul((average_c0-average_c1).T, covmat_inv)
b = -0.5*((average_c0.T).dot(covmat_inv).dot(average_c0)
         -(average_c1.T).dot(covmat_inv).dot(average_c1))+math.log(numData_c0/numData_c1)
print(-w,-b)
np.save('model.npy', np.hstack((w, b)))

########## Generative Model : testing (Ein) ##########
train_Yhat = np.matmul(train_X, w)+b
threshold = 0
train_Yhat[train_Yhat >  threshold] = 0
train_Yhat[train_Yhat < threshold] = 1
diff = train_Yhat-train_Y;
print('#total:', train_Yhat.size, '#wrong:', diff[diff!=0].size)
print('#class0(real):', train_Y[train_Y==0].size)
print('#class0(pred):', train_Yhat[train_Yhat==0].size)
