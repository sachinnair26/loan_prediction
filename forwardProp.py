import numpy as np 
from sigmoid import sigmoid

def forwardProp(X,theta1,theta2):
    m = np.shape(X)[0]
    a1 = np.append(X,np.ones((m,1)),axis=1)
    a2 = sigmoid(np.dot(a1,np.transpose(theta1)))
    a2 = np.append(np.ones((m,1)),a2,axis=1)
    a3 = sigmoid(np.dot(a2,np.transpose(theta2)))
    return a3
    # print(np.shape(a2))
    