#hypothesi = theta0 + theta1*sigma


import numpy as np
import random
import matplotlib.pyplot as plt
import math
import csv


# m denotes the number of examples here, not the number of features
def gradientDescent(x, y,stdDev, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)#change........
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f | theta: %f ,%f" % (i, cost,theta[0],theta[1]))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta



def genData(numPoints,a1,a2,stDev):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = stDev
        # our target variable
        y[i] = a2[i]#(i + bias) + random.uniform(0, 1) * variance
    return x, y

n=0
N=0
a1=[]
a2=[]
index=[]
stDev=0.0
with open('data.csv', "rb") as f1:
    f1 = [x for x in f1 ]
    for line in f1:

        data = line.split(",")
        if data[4] == 'Not Available':
    		break
        N=N+1
        #a1.append(int(data[0]))
        a1.append(int(data[4]))
        a2.append(int(data[5]))
        index.append(N)
stDev=np.std(a1)


x, y = genData(N,a1,a2,stDev)
#print(x)
#print(y)
m, n = np.shape(x)
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
print(theta)
theta = gradientDescent(x, y,stDev, theta, alpha, m, numIterations)
print(theta)
error=0
err=0
for i in range(0, N):
	if(a1[i]>a2[i]+theta[0]+theta[1]*stDev):
		error=error+1;
		err = err + a1[i]-a2[i]-theta[0]-theta[1]*stDev

print(error)
print(err)



plt.plot(index,a1,'b',index,a2,'g',index,a2 + np.dot(x, theta),'r')
plt.show()

