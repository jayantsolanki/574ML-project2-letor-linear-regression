#main.py
import numpy as np
import math
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from lib import *
K=3#number of clusters
L2_lambda=0.1
syn_input_data = np.loadtxt('input.csv', delimiter=',')
syn_output_data = np.genfromtxt('output.csv').reshape([-1, 1])
[N, D]=syn_input_data.shape
# RANDOMLY SHUFFLE THE DATA BEFORE PERFORMING THE KMEANS CLUSTERING
# letor_input_data = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')
# letor_output_data = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])
# print (syn_input_data.shape)
# print (syn_output_data.shape)

[centroids, spreads]=kMeans(syn_input_data, K)# number of clusters is 3
# centroids=centroids.reshape([K,1,D])
centroids = centroids[:, np.newaxis, :]
syn_input_data = syn_input_data[np.newaxis, :, :]
print (syn_input_data.shape)
# print ("Printing the centroids found:")
# print(centroids.shape)
lowest=100000
LAMBDA=1000
design_matrix=design_matrix(syn_input_data, centroids, spreads)
# calculating the closed form
for m in np.arange(0, 2,0.015):
	W=closed_form_sol(m, design_matrix, syn_output_data)
	W=W.reshape([K+1,1])
	# print (W)
	Y_dash = design_matrix.dot(W)
	Error =  np.sum(np.square(syn_output_data - Y_dash))/2 + 0.5*m*(W.T.dot(W))
	Erms = math.sqrt((2 * Error)/N)
	if lowest>Erms:
		lowest=Erms
		LAMBDA=m
	print("for lambda = %0.4f, ERMS = %0.4f"%(m, Erms))

print("Min Erms is = %0.4f " %lowest)
print("Min lambda is = %0.4f " %LAMBDA)