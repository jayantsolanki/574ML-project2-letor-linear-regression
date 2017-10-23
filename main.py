#main.py
import numpy as np
import math
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from lib import *
import matplotlib.pyplot as plt

K=20#number of clusters
L2_lambda=0.1
mode = 2


#np.random.shuffle(syn_input_data)

if mode == 1:
	X = np.loadtxt('input.csv', delimiter=',')
	Y = np.genfromtxt('output.csv').reshape([-1, 1])
elif mode == 2:
	X = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')
	Y = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])

[N, D]  = X.shape
X_train = X[0:int(0.8*N),:]
X_Val 	= X[int(0.8*N):int(0.9*N),:]
X_test  = X[int(0.9*N):N,:]
Y_train = Y[0:int(0.8*N),:]
Y_val 	= Y[int(0.8*N):int(0.9*N),:]
Y_test  = Y[int(0.9*N):N,:]
# RANDOMLY SHUFFLE THE DATA BEFORE PERFORMING THE KMEANS CLUSTERING
# letor_input_data = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')
# letor_output_data = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])
# X_train = letor_input_data[0:int(0.8*N),:]
# X_Val = letor_input_data[int(0.8*N):int(0.9*N),:]
# X_test = letor_input_data[int(0.9*N):N,:]
# Y_train = letor_input_data[0:int(0.8*N),:]
# Y_val = letor_input_data[int(0.8*N):int(0.9*N),:]
# Y_test  = letor_input_data[int(0.9*N):N,:]
# print (syn_input_data.shape)
# print (syn_output_data.shape)
# print (X_train.shape)
# print (X_Val.shape)
# print (X_test.shape)
# print (Y_train.shape)
# print (Y_val.shape)
# print (Y_test.shape) 

[centroids, spreads]=kMeans(X_train, K)# number of clusters is 3
# centroids=centroids.reshape([K,1,D])
centroids = centroids[:, np.newaxis, :]
X_train = X_train[np.newaxis, :, :]
print (X_train.shape)
# print ("Printing the centroids found:")
# print(centroids.shape)
lowest=100000
LAMBDA=1000
design_matrix_train=design_matrix(X_train, centroids, spreads)
design_matrix_val = design_matrix(X_Val, centroids, spreads)
design_matrix_test = design_matrix(X_Test, centroids, spreads)
# W_SGD = sgd_solution(learning_rate=1, minibatch_size=N, num_epochs=200, L2_lambda=0.1, design_matrix=design_matrix, output_data=syn_output_data, design_matrix_val, Y_Val)
# print(W)

###############Validation and Parameters fine tuning
# calculating the closed form
# Erms_train=0;
# l=0;
Erms_train = []
Erms_val = []
lam=[]
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for l in np.arange(0.001,1,0.001):
	W_CF=closed_form_sol(l, design_matrix_train, Y_train)
	W_CF=W_CF.reshape([K+1,1])
	# print (W_CF)
	ErmsTrain= erms(design_matrix_train, Y_train, W_CF, l)
	Erms_train.append(ErmsTrain)
	lam.append(l)
	ErmsVal = erms(design_matrix_val, Y_val, W_CF, l)
	Erms_val.append(ErmsVal)
	#Y_dash_test = design_matrix_val.dot(W)

	if lowest>ErmsVal:
		lowest=ErmsVal 	
		LAMBDA=l
	# print("for lambda = %0.4f, ERMS Train = %0.6f, ERMS Val = %0.6f"%(l, Erms_train, Erms_val))
# print(lam)
# print(Erms_train)
# axes = plt.gca()
# axes.set_ylim([0,1])
ax1.plot(np.log(lam), Erms_train, 'b-')
plt.ylabel('Training ERMS)$')
plt.xlabel('ln(λ))')
ax2.plot(np.log(lam), Erms_val, 'r-')
plt.ylabel('Validation ERMS')
plt.xlabel('ln(λ)')
# ax3.plot(np.log(lam), Erms_test, 'g-')
# plt.ylabel('Test ERMS')
# plt.xlabel('ln(λ)')
print("Min Erms in Valdiation Set is = %0.4f " %lowest)
print("At lambda is = %0.4f " %LAMBDA)
plt.show()
