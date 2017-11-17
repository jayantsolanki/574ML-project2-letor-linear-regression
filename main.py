#main.py
import numpy as np
import math
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from lib import *
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

#K=20#number of clusters
#L2_lambda=0.1
data_mode = 2
implementation_mode = 'C' #C - Closed form solution; S- Stochastic Gradient Descent

if data_mode == 1:
	X = np.loadtxt('input.csv', delimiter=',')
	Y = np.genfromtxt('output.csv').reshape([-1, 1])
elif data_mode == 2:
	X = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')
	Y = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])


# X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state=0)
# X_Val, X_Test, Y_Val, Y_Test = train_test_split(X_Test, Y_Test, test_size = 0.5, random_state=0)
[N, D]  = X.shape
X_Train = X[0:int(0.8*N),:]
X_Val 	= X[int(0.8*N):int(0.9*N),:]
X_Test  = X[int(0.9*N):N,:]
Y_Train = Y[0:int(0.8*N),:]
Y_Val 	= Y[int(0.8*N):int(0.9*N),:]
Y_Test  = Y[int(0.9*N):N,:]
# RANDOMLY SHUFFLE THE DATA BEFORE PERFORMING THE KMEANS CLUSTERING
print (X_Train.shape)
print (X_Val.shape)
print (X_Test.shape)
print (Y_Train.shape)
print (Y_Val.shape)
print (Y_Test.shape) 	


# W_SGD = sgd_solution(learning_rate=1, minibatch_size=N, num_epochs=200, L2_lambda=0.1, design_matrix=design_matrix, output_data=syn_output_data, design_matrix_val, Y_Val)

###############Validation and Parameters fine tuning
# calculating the closed form
# Erms_train=0;
# l=0;

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
k=80
while k<=80:
	try:
		[centroids, spreads]=kMeans(X_Train, k)# number of clusters is 3
	except:
		continue

	# centroids=centroids.reshape([K,1,D])
	centroids = centroids[:, np.newaxis, :]
	X_Train2 = X_Train[np.newaxis, :, :]
	#print (X_Train.shape)
	# print ("Printing the centroids found:")
	# print(centroids.shape)
	lowest=100000
	LAMBDA=1000

	design_matriX_Train=design_matrix(X_Train2, centroids, spreads)
	design_matriX_Val = design_matrix(X_Val, centroids, spreads)
	design_matriX_Test = design_matrix(X_Test, centroids, spreads)
	# print(design_matriX_Train.shape)
	Erms_train = []
	Erms_val = []
	Erms_test = []
	lam=[]
	W_star=[]
	trainError=0
	for l in np.arange(0.1,2,0.1):
		if implementation_mode == 'C':
			W = closed_form_sol(l, design_matriX_Train, Y_Train)
		elif implementation_mode == 'S':
			W = sgd_solution(learning_rate=1, minibatch_size=5569, num_epochs=100, L2_lambda=0.1, design_matrix=design_matriX_Train, output_data=Y_Train, design_matrix_val = design_matriX_Val, Y_Val = Y_Val)
		W = W.reshape([k+1,1])
		# print (W.shape)
		# break
		# print (W_CF)
		ErmsTrain= erms(design_matriX_Train, Y_Train, W, l)
		Erms_train.append(ErmsTrain)
		lam.append(l)
		ErmsVal = erms(design_matriX_Val, Y_Val, W, l)
		Erms_val.append(ErmsVal)
		#Y_dash_test = design_matriX_Val.dot(W)

		# if lowest>ErmsVal:
		# 	lowest=ErmsVal 	
		# 	LAMBDA=l
		# 	break
		if lowest>ErmsVal:
			lowest=ErmsVal 	
			trainError = ErmsTrain
			LAMBDA=l
			W_star=W
	ErmsTest = erms(design_matriX_Test, Y_Test, W_star, LAMBDA)
	Erms_test.append(ErmsTest)
	# print("At lambda : %0.4f and K = %d" %(LAMBDA, k))
	# print("		Min Erms in Training Set is = %0.4f" %trainError)
	# print("		Min Erms in Validation Set is = %0.4f" %lowest)
	# print("		Error in Test Set is = %0.4f" %ErmsTest)
	print("%d %0.4f %0.4f %0.4f %0.4f" %(k, LAMBDA, trainError, lowest, ErmsTest))
	k = k+5
	# print("for lambda = %0.4f, ERMS Train = %0.6f, ERMS Val = %0.6f"%(l, Erms_train, Erms_val))
# print(lam)
# print(Erms_train)
# axes = plt.gca()
# axes.set_ylim([0,1])
# ax1.plot(np.log(lam), Erms_train, 'b-')
# plt.ylabel('Training ERMS')
# plt.xlabel('ln(λ))')
# ax2.plot(np.log(lam), Erms_val, 'r-')
# plt.ylabel('Validation ERMS')
# plt.xlabel('ln(λ)')
# # ax3.plot(np.log(lam), Erms_test, 'g-')
# # plt.ylabel('Test ERMS')
# # plt.xlabel('ln(λ)')
# plt.show()

#Plot graph
# plt.subplot(1, 1, 1)
# plt.plot(np.log(lam), Erms_train, label='Training')
# plt.plot(np.log(lam), Erms_val, label='Validation')
# #plt.vlines(alpha_optim, plt.ylim()[0], np.max(Erms_val), color='k', linewidth=3, label='Optimum on test')
# plt.legend()
# plt.xlabel('ln(λ)')
# plt.ylabel('Root Mean Square Error')
# plt.show()