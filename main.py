import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forwardProp import forwardProp
from backwardProp import backwardProp
from costFunction import costFunction
from sklearn.neural_network import MLPClassifier
X = pd.read_csv('./train_data.csv')
X = X.drop(labels='Loan_ID',axis=1)
X = pd.DataFrame(X)
X.dropna(inplace=True)

y = X.loc[:,'Loan_Status']



# comparison between Marrige and Loan
a = pd.crosstab(X['Married'],y)
plt.xlabel('Married')
plt.ylabel('Count')
plt.bar(['No','Yes'],a['Y'].values)
plt.bar(['No','Yes'],a['N'].values)
plt.legend(['Got Loan','Didnt get Loan'])
plt.show()

a = pd.crosstab(X['Gender'],y)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.bar(['Female','Male'],a['Y'].values)
plt.bar(['Female','Male'],a['N'].values)
plt.legend(['Got Loan','Didnt get Loan'])
plt.show()

a = pd.crosstab(X['Self_Employed'],y)
plt.xlabel('Self Employed')
plt.ylabel('Count')
plt.bar(['No','Yes'],a['Y'].values)
plt.bar(['No','Yes'],a['N'].values)
plt.legend(['Got Loan','Didnt get Loan'])
plt.show()

a = pd.crosstab(X['Education'],y)
plt.xlabel('Education')
plt.ylabel('Count')
plt.bar(['Graduate','Not Graduate'],a['Y'].values)
plt.bar(['Graduate','Not Graduate'],a['N'].values)
plt.legend(['Got Loan','Didnt get Loan'])
plt.show()


a = pd.crosstab(X['Credit_History'],y)
plt.xlabel('Credit History')
plt.ylabel('Count')
plt.bar(['0','1'],a['Y'].values)
plt.bar(['0','1'],a['N'].values)
plt.legend(['Got Loan','Didnt get Loan'])
plt.show()



a = pd.crosstab(X['Property_Area'],y)
plt.xlabel('Property Area')
plt.ylabel('Count')
plt.bar(['Rural','Semi Urban','Urban'],a['Y'].values)
plt.bar(['Rural','Semi Urban','Urban'],a['N'].values)
plt.legend(['Got Loan','Didnt get Loan'])
plt.show()

t = X.corr()
fig, ax = plt.subplots()
ax.imshow(t, cmap='hot')
ax.set_xticks([0,1,2,3,4,5,6,7])
ax.set_yticks([0,1,2,3,4,5,6,7])
ax.set_xticklabels(t.columns)
ax.set_yticklabels(t.columns)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Heat Map For various properties")
plt.show()

X['Loan_Status'].replace('Y',1,inplace=True)
X['Loan_Status'].replace('N',0,inplace=True)
X.drop('Loan_Status',axis=1)

y = np.asmatrix(y).T
X['Married'].replace('Yes',1,inplace=True)
X['Married'].replace('No',0,inplace=True)
X['Gender'].replace('Male',1,inplace=True)  
X['Gender'].replace('Female',0,inplace=True)
X['Education'].replace('Graduate',1,inplace=True)
X['Education'].replace('Not Graduate',0,inplace=True)
X['Self_Employed'].replace('Yes',1,inplace=True)
X['Self_Employed'].replace('No',0,inplace=True)
X['Property_Area'].replace('Rural',1,inplace=True)
X['Property_Area'].replace('Semiurban',2,inplace=True)
X['Property_Area'].replace('Urban',3,inplace=True)
X.drop('Loan_Status',axis=1,inplace=True)
X.drop('Dependents',axis=1,inplace=True)
X_train = X.iloc[:400,:]
X_test = X.iloc[400:,:]
y_train = y[:400,:]
y_test = y[400:,:]
size = np.shape(X_train)
m = size[0]
input_layer_size = size[1]
hidden_layer_size = 5 
final_layer_size = 1   
lamb =3
alpha = 0.1
cost = []
X_train = np.array(X_train)
theta1 = np.random.rand(hidden_layer_size,input_layer_size+1)
theta2 = np.random.rand(final_layer_size,hidden_layer_size+1)
for j in range(0,2500):
    newtheta1,newtheta2 = backwardProp(X_train,y_train,theta1,theta2,lamb)
    theta1 = theta1 - alpha*newtheta1
    theta2 = theta2 - alpha*newtheta2
    a3 = forwardProp(X_train,theta1,theta2)
    J = costFunction(a3,y_train,theta1,theta2,lamb)
    cost.append(J)
plt.plot(cost)
plt.show()

a_test_3 = forwardProp(X_test,theta1,theta2)


# a_test_3 = pd.DataFrame(a_test_3)
# c = a_test_3[0].apply(lambda x:1 if x >=0.5 else 0 )
# y_test = pd.DataFrame(y_test)
# val = np.equal(c.values,y_test[0].values)
# counter = 0
# for p in range(0,80):
#     if val[p] == True:
#         counter = counter+1

# print(counter)


pred = np.argmax(a_test_3,axis=1)+1
T = np.equal(pred,y_test)
T = pd.DataFrame(T)
T.replace(True,1,inplace=True)
T.replace(False,0,inplace=True)
print('accurecy is ',(sum(T.values)/80)*100,'my nn')


y_train = pd.DataFrame(y_train)
y_train = y_train[0].values
nn = MLPClassifier(solver='lbfgs', alpha=0.0001,hidden_layer_sizes=(5),max_iter=1000)
nn.fit(X_train,y_train)
r = nn.predict(X_test)
y_test = pd.DataFrame(y_test)
y_test = y_test[0].values
T = np.equal(r,y_test)
T = pd.DataFrame(T)
T.replace(True,1,inplace=True)
T.replace(False,0,inplace=True)
print('accurecy is ',(sum(T.values)/80)*100,'by sklearn')