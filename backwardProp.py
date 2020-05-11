import numpy as np 
from sigmoid import sigmoid

def backwardProp(X,y,theta1,theta2,lamb):
    m = np.shape(X)[0]
    a1 = np.append(np.ones((m,1)),X,axis=1)
    a2 = sigmoid(np.dot(a1,np.transpose(theta1)))
    a2 = np.append(np.ones((m,1)),a2,axis=1)
    a3 = sigmoid(np.dot(a2,np.transpose(theta2)))
    del3 = a3 - y
    del2 = np.multiply(np.dot(del3,theta2),np.multiply(a2,(1-a2)))
    del2 = del2[:,1:]
    newtheta1 = np.dot(del2.T,a1)/m
    newtheta2 = np.dot(del3.T,a2)/m
    newtheta1[:,1:] = newtheta1[:,1] + (lamb/m)*theta1[:,1:]
    newtheta2[:,1:] = newtheta2[:,1] + (lamb/m)*theta2[:,1:]
    return newtheta1,newtheta2