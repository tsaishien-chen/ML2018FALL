import numpy as np
import random
import csv
import math

datain_X = np.genfromtxt('train_x.csv', delimiter=',')
datain_X = datain_X[1:,:]
datain_Y = np.genfromtxt('train_y.csv', delimiter=',')
datain_Y = datain_Y[1:]

################# input data cleaning ################
def cleandata(x):
    datasize = x.shape[0]
    
    # data preprocess   
    pre = np.ones(datasize)*1.1
    for i in range(datasize):
        PA = x[i,5:11]
        BA = x[i,11:17]
        b1 = ((PA[0]>=2) or (np.sum(PA)>10 and np.sum(PA)<17))
        b2 = (BA[0]/x[i,0]>1.5 and PA[PA>=2].size>0)
        
        if (not (b1 or b2)):
            pre[i] *= -1
       
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
 
    x = np.hstack((np.ones((datasize, 1)), x))
    return x, pre
    
datain_X, datain_pre = cleandata(datain_X)
datasize, datalength = datain_X.shape

################# get validation data ################
numValid = int(datasize/10)
random.seed(0)
randomlist = list(range(datasize))
random.shuffle(randomlist)
valid_X = datain_X[randomlist[:numValid],:]
valid_Y = datain_Y[randomlist[:numValid]]
train_X = datain_X[randomlist[numValid:],:]
train_Y = datain_Y[randomlist[numValid:]]
datasize, datalength = train_X.shape

########### Logistic regression : training ###########
w         = np.array([-1.33271468,-7.82523457e-03,-1.37746465e-05, 9.93500277e-01, 2.11048861e-03,
                                   1.37702804e-03, 1.60907318e-03, 1.23333731e-03, 1.03480095e-03,
                                   1.21938314e-02,-2.37236810e-02,-6.61893777e-02,-9.98015477e-05,
                                   0.00000000e+00, 3.30498002e-01, 3.30562621e-01, 3.30819443e-01,
                                  -6.55864510e-01,-6.58454030e-01, 3.22438474e-01])
iteration = 800
ir        = 10**(-7)
preErr    = numValid
gradLen   = 0

for i in range(iteration):
    z        = np.matmul(train_X, w)
    theta    = 1/(1+np.exp(-z))
    gradient = np.matmul(train_X.T, (theta-train_Y))/datasize
    gradLen += np.sum(gradient**2)
    w       -= gradient*(ir/math.sqrt(gradLen))
    
    if(i%20 == 19):
        z      = np.matmul(valid_X, w)
        z[z>0] = 0; z[z<0] = 1
        diff   = valid_Y-z;
        err    = diff[diff!=0].size
        if (i%100 == 99):
            print("iter:", i, "; loss:", err)
            
        if (err > preErr):
            break
        else:
            preErr = err
print(w)
np.save('model_best.npy', w)

######### Logistic regression : testing (Ein) #########
datain_Yhat = (np.matmul(datain_X, w).flatten()+datain_pre)
threshold = 0
datain_Yhat[datain_Yhat >  threshold] = 1
datain_Yhat[datain_Yhat <= threshold] = 0
diff = datain_Yhat-datain_Y;
print('#total:', datain_Yhat.size, '#wrong:', diff[diff!=0].size)
print('#class0(real):', datain_Y[datain_Y==0].size)
print('#class0(pred):', datain_Yhat[datain_Yhat==0].size)
