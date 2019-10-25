import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv


data = pd.read_csv("~/Documents/mnist.csv")
data = np.array(data)
a = data.transpose()
print(a[:,1].shape)
b = np.ones((2,))
c = np.ones((2,1))
print(b.ndim)
print(c.ndim)
x = np.array([1,2,3,4,5])
print(x.ndim)
z = np.zeros((2,2,2))
print(z[0].ndim)
y = np.expand_dims(x, axis = 1)
print(y.ndim)
print(y.shape)
u = np.array([[1],[4]])
mat = np.array([[10,11],[12,13]])
t = u.flatten()
mat[:,0] = t
print(mat[:,1] == mat[:,0])
