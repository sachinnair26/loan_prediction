import numpy as np 
from sigmoid import sigmoid

def costFunction(A,y,theta1,theta2,lamb):
    m = np.shape(A)[0]
    return -(1/m)*np.sum(np.multiply(y,np.log(A))+np.multiply((1-y),np.log(1-A)))+lamb*(np.sum(np.sum(np.multiply(theta1[:,2:],theta1[:,2:])))+np.sum(np.sum(np.multiply(theta2[:,2:],theta2[:,2:]))))/(2*m)
