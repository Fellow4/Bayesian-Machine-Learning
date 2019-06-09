import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
from numpy.linalg import inv as inverse
from numpy.linalg import det

def Standard(x,slope,intercept):
    return slope * x + intercept

def pos_deviation(x,a,b,c,d,slope,intercept):
    return slope *x + intercept + pow(1 + a + (b+c)*x + d*x*x,0.5)

def neg_deviation(x,a,b,c,d,slope,intercept):
    return slope *x + intercept - pow((1 + a + (b+c)*x + d*x*x),0.5)

#Plot for some feature at position in csv file given by index
def plot(data,index,target):
   one = np.ones((data.shape[0],1),dtype = float)
   train = np.concatenate((one,data[:,[index]]),axis = 1)
   alpha,beta = 1,1
   I = np.identity(train.shape[1],dtype = float)
   cov = inverse(alpha*I + beta*(np.matmul(train.transpose(),train)))
   Mean = beta*np.matmul(cov,np.matmul(train.transpose(),target))
   slope = Mean[1][0]
   intercept = Mean[0][0]
   a,b,c,d = cov[0][0],cov[0][1],cov[1][0],cov[1][1]
   x = np.linspace(1,1000,500)
   plt.xlabel('x')
   plt.ylabel('prediction')
   plt.title('posterior predictive inferences')
   plt.plot(x,Standard(x,slope,intercept),color = 'g')
   plt.plot(x,pos_deviation(x,a,b,c,d,slope,intercept),color = 'r')
   plt.plot(x,neg_deviation(x,a,b,c,d,slope,intercept),color = 'r')
   plt.show()

#Bayesain Cost Prediction
def prediction(data,w):
    cost = np.array(np.matmul(data,w))
    return cost

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#Distribution for a value of y given input features and the weight matrix
def predictive_distribution(covar,mean,beta,samples):
    for i in range (samples):
     input = np.array(temp[[i],:]).transpose()
     Mean = np.matmul(mean.transpose(),input)[(0,0)]
     Variance = 1.0/beta + np.matmul(np.matmul(input.transpose(),covar),input)
     sigma = np.sqrt(Variance)[0,0]
     x_values = np.linspace(3,50,10000)
     plt.xlabel("x")
     plt.ylabel("Probability")
     plt.title("Probability Distribution Curves")
     plt.plot(x_values,gaussian(x_values,Mean,sigma),linewidth = 2,color = 'b')
    plt.show()
    return

#Root Mean Square Error
def Error(predicted,actual):
    N = actual.shape[0]
    sum = np.matmul((predicted-actual).transpose(),(predicted-actual))[(0,0)]
    RMS = math.sqrt((2.0/N)*sum)
    return RMS

#Trains a Polynomial Regression Classifier and outputs the RMS Error for the chosen value of k
def Polynomial_Regression(k,input,target,test_data,y_test):
    #Create features to capture powers upto k
    phi = np.ones((input.shape[0],1),dtype = float)
    for i in range(1,k+1):
        phi = np.concatenate((phi,np.power(input,i)),axis = 1)
    alpha,beta = 1,1
    I = np.identity(k+1,dtype = float)
    covar = inverse(alpha*I + beta*(np.matmul(phi.transpose(),phi)))
    mean = beta*np.matmul(covar,np.matmul(phi.transpose(),target))
    ans = []
    for i in range(test_data.shape[0]):
        sum = 0
        for j in range(k+1):
            sum = sum + (math.pow(test_data[i][0],j))*mean[j][0]
        ans.append(sum)
    ans = np.array(ans)
    ans = ans.transpose()
    deviation = Error(ans,y_test)
    return deviation

data =  pd.read_csv("~/Documents/BostonHousing.csv")
test = pd.read_csv("~/Documents/test.csv")
test = np.array(test)
row = test.shape[0]
temp = np.delete(test,0,axis = 1)
temp = np.concatenate((np.ones((row,1),dtype = float),temp),axis = 1)
alpha,beta = 10,1
mat = np.array(data)
cols = mat.shape[1]
target = mat[:,[cols-1]]
phi = np.delete(mat,cols-1,axis = 1)
rows = phi.shape[0]
y_train = target[0:335]
y_test = target[335:rows]
one = np.ones((rows,1),dtype = float)
phi = np.concatenate((one,phi),axis = 1)
train = phi[0:335]
test_data = phi[335:rows]
I = np.identity(train.shape[1],dtype = float)
S_N = inverse(alpha*I + beta*(np.matmul(train.transpose(),train)))
M_N = beta*np.matmul(S_N,np.matmul(train.transpose(),y_train))
bayesian_cost = np.array(prediction(test_data,M_N))
#Plots for Predictive Distribution for first 5 outputs
#predictive_distribution(S_N,M_N,beta,5
RMS = Error(bayesian_cost,y_test)
print("RMS Error is:",RMS)
#Weights obtained by the frequentist approach
#Moore Penrose inverse
w = np.matmul(np.matmul(inverse(np.matmul(phi.transpose(),phi)),phi.transpose()),target)
frequentist_cost = np.array(prediction(temp,w))

#Plot for a feature zn(2nd feature)
plot(train,2,y_train)

#Polynomial Regression
Error_poly = Polynomial_Regression(3,train[:,[2]],y_train,test_data,y_test)
print("Error corresponding to poly regression:",Error_poly)
