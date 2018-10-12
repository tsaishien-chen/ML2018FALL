import numpy as np
import scipy as sp
import csv

train = np.genfromtxt('train.csv', delimiter=",")
train = train[1:, 3:]
train[np.isnan(train)] = 0

parameters, hours = 18, 9
datalength = 3*hours
days, months = 20, 12
datasize = (24*days-hours)*months

def getdata(index):
    month   = int(index/(24*days-hours))
    index   = index%(24*days-hours)
    day     = int(index/24)
    hour    = index%24
    row     = (month*days+day)*parameters
    nextrow = row+parameters
    
    data = np.empty([3,hours])
    ans  = 0
    if (hour < 24-hours):
        data = np.vstack((train[row+2,hour:hour+hours],train[row+8:row+10,hour:hour+hours]))
        ans  = train[row+9,hour+hours]
    elif (hour == 24-hours):
        data = np.vstack((train[row+2,hour:],train[row+8:row+10,hour:]))
        ans  = train[nextrow+9,0]
    else:
        data = np.hstack((np.vstack((train[row+2,hour:],train[row+8:row+10,hour:])),
                          np.vstack((train[nextrow+2,:hour+hours-24],train[nextrow+8:nextrow+10,:hour+hours-24]))))
        ans  = train[nextrow+9,hour+hours-24]
           
    CO          = data[0]
    CO_less     = (CO[CO<=0].size==0)
    pm10        = data[1]
    pm10_less   = (pm10[pm10<=0].size==0)
    pm25        = data[2]
    pm25_less   = (ans>0 and pm25[pm25<=0].size==0)
    pm25_larger = (ans<=111 or abs(ans-pm25[hours-1])<30) and (pm25[pm25>100].size==0 or np.max(np.abs(np.diff(pm25)))<30)
    valid = CO_less and pm10_less and pm25_less and pm25_larger
    
    return (data.flatten(), ans, valid)

x = np.ones([datasize,datalength+1])
y = np.empty(datasize)
validcount = 0
for i in range(datasize):
    x[validcount,1:], y[validcount], valid = getdata(i)
    if (valid):
        validcount += 1
x = np.delete(x, list(range(validcount,datasize)), axis=0)
y = np.delete(y, list(range(validcount,datasize)))

regularization = 0
pseudoinv = np.matmul(np.linalg.inv(np.matmul(x.T,x)+regularization*np.identity(datalength+1)),x.T)
w = np.matmul(pseudoinv,y.reshape((validcount,1))).flatten()
print(w)

errorcount = 0
diff = np.empty(datasize)
for i in range(datasize):
    xvec, yhat, valid = getdata(i)
    if (valid):
        y = np.dot(xvec, w[1:])+w[0]
        diff[i] = abs(yhat-y)
        if (diff[i] > 12):
            print(errorcount, i, diff[i], xvec[2*hours+5:], yhat, y)
            errorcount += 1
    else:
        diff[i] = 0
print('#all data:', datasize, '; #valid data:', validcount)
print("Ein:", sp.sqrt(np.sum(diff**2)/validcount))

test = np.genfromtxt('test.csv', delimiter=",")
test = test[:,2:]
test[np.isnan(test)] = 0

y = np.empty(260)
for i in range(260):
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

with open('testresult14.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','value'])
    for i in range(260):
        writer.writerow([('id_'+str(i)),y[i]])
