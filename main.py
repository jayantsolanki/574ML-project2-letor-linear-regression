#main.py
import numpy as np
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
# letor_input_data = np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
# letor_output_data = np.genfromtxt('datafiles/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])
# print (syn_input_data.shape)
# print (syn_output_data.shape)

[centroids, spreads]=kMeans(syn_input_data, K)# number of clusters is 3
# centroids=centroids.reshape([K,1,D])
centroids = centroids[:, np.newaxis, :]
syn_input_data = syn_input_data[np.newaxis, :, :]
print (syn_input_data.shape)
# print ("Printing the centroids found:")
# print(centroids.shape)
design_matrix=design_matrix(syn_input_data, centroids, spreads)
W=closed_form_sol(L2_lambda, design_matrix, syn_output_data)
print(W.T.)
