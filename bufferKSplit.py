#hypothesi = theta0 + theta1*sigma


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import csv
from sklearn.cross_validation import KFold  


# m denotes the number of examples here, not the number of features
def gradientDescent(x, y,stdDev, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)#change........
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta



def genData(no1,no2,N,a1,a2,stDev):
    numPoints=no1-N-1
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0,numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = stDev
        # our target variable
        y[i] = a2[i]#(i + bias) + random.uniform(0, 1) * variance
    return x, y

n=0
N=1
no1=0
no2=0
index=[]
stDev=0.0
pid='A'
a1=[]
a2=[]
res=[]
count=0
error=0
err=0
numIterations= 10000
alpha = 0.0005
stDev=0.0
StDev=0.0
with open('data.csv', "rb") as f1:
    f1 = [x for x in f1]
    for line in f1:

        data = line.split(",")
        count=count+1
        #print pid
        #print data[1]
        if data[4]=='Not Available':
            no2 = no2+1
            a1.append(int(0))
            a2.append(int(0))
            index.append(no1+no2)
        elif pid==data[1]:
            no1=no1+1
            a1.append(int(data[4]))
            a2.append(int(data[5]))
            index.append(no1+no2)
        else:
            min_err=100000000000000000
            pid=data[1]
            a1.append(int(data[4]))
            a2.append(int(data[5]))
            index.append(no1+no2)
            no1=no1+1
            x, y = genData(no1-1,no2,N,a1,a2,stDev)
            m, n = np.shape(x)
            theta = np.ones(n)
            kf = KFold(len(x), n_folds=4)


            for train, test in kf:
                x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
                #print("%s %s" % (train, test))
                #print("%s %s" % (x_train,y_test))
                stDev=np.std(y_train)
                theta = gradientDescent(x_train, y_train,stDev, theta, alpha, m, numIterations)
                #print(theta)
                error=0
                err=0
                StDev=np.std(y_test)
                for i in range(0, len(x_test)):
                    if(a1[test[i]]>a2[test[i]]+theta[0]+theta[1]*StDev):
                        error=error+1
                        err = err + a1[test[i]]-a2[test[i]]-theta[0]-theta[1]*StDev
                #print error
                #print err
                if err<min_err:
                    thetaMin=theta
                    min_err = err
                minDev=stDev



            #stDev=np.std(a1)
            
            
            theta=thetaMin
            print(theta)
            for v in range(N,no1+no2):
                res.append(math.ceil(theta[0]+theta[1]*stDev))
            N=no1+no2
            no1=N
            no2=0
            
        
            



res.append(0)    
np.savetxt("bufferKsplit.csv", res, delimiter=",")
print len(res)
print len(a2)
res2=[]
for v in range(0,len(a2)):
    res2.append(res[v]+a2[v])
plt.plot(index,a1,'b',index,res2,'r')

            
                
        
            



plt.show()

