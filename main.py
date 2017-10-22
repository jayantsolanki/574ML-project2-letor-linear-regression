#main.py
import numpy as np
import math
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from lib import *
import matplotlib.pyplot as plt

K=3#number of clusters
L2_lambda=0.1
syn_input_data = np.loadtxt('input.csv', delimiter=',')
syn_output_data = np.genfromtxt('output.csv').reshape([-1, 1])
[N, D]=syn_input_data.shape
np.random.shuffle(syn_input_data)
X_train = syn_input_data[0:int(0.8*N),:]
X_Val = syn_input_data[int(0.8*N):int(0.9*N),:]
X_test = syn_input_data[int(0.9*N):N,:]
Y_train = syn_output_data[0:int(0.8*N),:]
Y_val = syn_output_data[int(0.8*N):int(0.9*N),:]
Y_test  = syn_output_data[int(0.9*N):N,:]
# RANDOMLY SHUFFLE THE DATA BEFORE PERFORMING THE KMEANS CLUSTERING
# letor_input_data = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')
# letor_output_data = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])
# print (syn_input_data.shape)
# print (syn_output_data.shape)
print (X_train.shape)
print (X_Val.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_val.shape)
print (Y_test.shape)
# [centroids, spreads]=kMeans(X_train, K)# number of clusters is 3
# # centroids=centroids.reshape([K,1,D])
# centroids = centroids[:, np.newaxis, :]
# X_train = X_train[np.newaxis, :, :]
# print (X_train.shape)
# # print ("Printing the centroids found:")
# # print(centroids.shape)
# lowest=100000
# LAMBDA=1000
# design_matrix_train=design_matrix(X_train, centroids, spreads)
# # W_SGD = sgd_solution(learning_rate=1, minibatch_size=N, num_epochs=200, L2_lambda=0.1, design_matrix=design_matrix, output_data=syn_output_data)
# # print(W)

# ###############Validation and Parameters fine tuning

# # calculating the closed form
# # Erms_train=0;
# # l=0;
# Erms_train = []
# Erms_val = []
# lam=[]
# fig = plt.figure()
# ax = fig.add_subplot(111)

# for l in np.arange(0.,0.5,0.00001):
# 	W_CF=closed_form_sol(l, design_matrix_train, Y_train)
# 	W_CF=W_CF.reshape([K+1,1])
# 	# print (W_CF)
# 	Erms_train.append(erms(design_matrix_train, Y_train, W_CF, l))
# 	lam.append(l)
# 	design_matrix_val = design_matrix(X_Val, centroids, spreads)
# 	Erms_val.append(erms(design_matrix_val, Y_val, W_CF, l))
# 	#Y_dash_test = design_matrix_val.dot(W)

# 	# if lowest>Erms:
# 	# 	lowest=Erms
# 	# 	LAMBDA=l
# 	# print("for lambda = %0.4f, ERMS Train = %0.6f, ERMS Val = %0.6f"%(l, Erms_train, Erms_val))
# # print(lam)
# # print(Erms_train)
# axes = plt.gca()
# axes.set_ylim([0,1])
# line1, = ax.plot(np.log(lam), Erms_train, 'b-')
# # ax2 = ax.twinx()
# # line1, = ax2.plot(np.log(lam), Erms_val, 'b-')
# # print("Min Erms is = %0.4f " %lowest)
# # print("Min lambda is = %0.4f " %LAMBDA)
# plt.show()
