import numpy as np
import sys
import csv

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
    
w = np.load('model.npy')

test_X = np.genfromtxt(sys.argv[3], delimiter=',')
test_X = test_X[1:,:]
test_X = cleandata(test_X)
test_X = np.hstack((test_X, np.ones((test_X.shape[0], 1))))
test_Y = np.matmul(test_X, w)
threshold = 0
test_Y[test_Y > threshold] = 0
test_Y[test_Y < threshold] = 1

with open(sys.argv[4], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','Value'])
    for i in range(test_Y.size):
        writer.writerow([('id_'+str(i)), int(test_Y[i])])
        
