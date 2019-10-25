import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import pinv
import csv

def Nk(gamma,n,idx):
    sum = 0
    for i in range(0,n):
        sum = sum + gamma[i][idx]
    return sum

def update_mu(data,gamma,n,mu,k):
    temp = data.transpose()
    for j in range(0,k):
        sum = np.zeros((784,1))
        for i in range(0,n):
            x = np.expand_dims(temp[:,i], axis = 1)
            sum = sum + gamma[i][j]*x
        val = Nk(gamma,n,j)
        t = sum/val
        mu[:,j] = t.flatten()
    return mu

def update_sigma(data,gamma,n,sigma,k):
    temp = data.transpose()
    for j in range(0,k):
        sum = np.zeros((784,784))
        for i in range(0,n):
            y = np.expand_dims(mu[:,j], axis = 1)
            x = np.expand_dims(temp[:,i], axis = 1)
            sum += gamma[i][j]*np.matmul(x-y,(x-y).transpose())
        val = Nk(gamma,n,j)
        sigma[j] = sum/val
    return sigma

def update_pi(gamma,n,k,pi):
    sum = 0
    for i in range (0,k):
        sum += Nk(gamma,n,i)
    for i in range (0,k):
        pi[i] = Nk(gamma,n,i)/sum
    return pi

def args(data,mu,sigma,n,k):
  temp = data.transpose()
  mat = np.ones((n,k))
  for i in range(0,n):
    for j in range(0,k):
        x = np.expand_dims(mu[:,j], axis = 1)
        y = np.expand_dims(temp[:,i], axis = 1)
        val = np.matmul(np.matmul((y-x).transpose(),pinv(sigma[j],rcond = 1e-15)),y-x)[0][0]
        mat[i][j] = -0.5 * val
  return mat

from numpy.linalg import det

def Det(sigma,k):
    mat = np.ones((k,1))
    for i in range(0,k):
        mat[i] = det(sigma[i])
    return mat

def g_n_k(data,args,mu,sigma,pi,n,k,dets):
   gamma = np.ones((n,k))
   for i in range (0,n):
    max = args[i][0]
    for j in range (0,k):
        if args[i][j] > max :
            max = args[i][j]
    temp = 0
    for j in range (0,k):
        temp = (pi[j]/math.sqrt(dets[j]))*math.exp(args[i][j]-max) + temp
    temp = max+math.log(temp)
    for j in range (0,k):
        gamma[i][j] = math.log(pi[j])-0.5*math.log(dets[j])+args[i][j] - temp
    gamma = np.exp(gamma)
    return gamma





data = pd.read_csv("~/Documents/mnist.csv")
data = np.array(data)
data = data[400:407]
n = data.shape[0]
k = 10
size = 784
s = 28
pi = 0.1*np.ones((k,1),dtype = 'float')
sigma = []
for i in range (0,k):
    sigma.append(np.identity(size))
sigma = np.asarray(sigma)
mu = np.random.rand(size,k)

for iterations in range(0,1):
    mat = args(data,mu,sigma,n,k)
    Dets = Det(sigma,k)
    gamma = g_n_k(data,mat,mu,sigma,pi,n,k,Dets)
    mu = update_mu(data,gamma,n,mu,k)
    sigma = update_sigma(data,gamma,n,sigma,k)
    pi = update_pi(gamma,n,k,pi)

for i in range(0,n):
    a = mu[:,i].reshape((28,28))
    plt.imshow(a,'gray',origin = 'lower')
    plt.show()
