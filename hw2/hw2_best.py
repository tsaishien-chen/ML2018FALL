import numpy as np
import sys
import csv

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
    
w = np.load('model_best.npy')

test_X = np.genfromtxt(sys.argv[3], delimiter=',')
test_X = test_X[1:,:]
test_X, test_pre = cleandata(test_X)
test_Yhat = (np.matmul(test_X, w).flatten()+test_pre)
threshold = 0
test_Yhat[test_Yhat > threshold] = 1
test_Yhat[test_Yhat < threshold] = 0

with open(sys.argv[4], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','Value'])
    for i in range(test_Yhat.size):
        writer.writerow([('id_'+str(i)), int(test_Yhat[i])])
        
