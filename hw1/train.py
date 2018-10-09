import numpy as np
import sys
import csv

test = np.genfromtxt(sys.argv[1], delimiter=",", encoding="big5")
test = test[:,2:]
test[np.isnan(test)] = 0

parameters, hours = 18, 9
datasize = int(test.shape[0]/parameters)
w = np.load('model.npy')

y = np.empty(datasize)
for i in range(datasize):
    row = i*parameters
    x = np.vstack((test[row+2],test[row+8:row+10]))

    for j in range(3):
        a = x[j]
        zero = np.where(a<=0)[0]
        while(zero.size!=0 and zero.size!=hours):
            for k in range(zero.size):
                if (zero[k]!=0 and a[zero[k]-1]>0):
                    a[zero[k]] = a[zero[k]-1]
                elif (zero[k]!=a.size-1 and a[zero[k]+1]>0):
                    a[zero[k]] = a[zero[k]+1]
            zero = np.where(a <= 0)[0]
    y[i] = np.dot(x.flatten(), w[1:])+w[0]

with open(sys.argv[2], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','value'])
    for i in range(datasize):
        writer.writerow([('id_'+str(i)),y[i]])
   
