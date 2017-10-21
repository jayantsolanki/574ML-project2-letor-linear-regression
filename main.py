#main.py
import numpy as np
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
syn_input_data = np.loadtxt('input.csv', delimiter=',')
syn_output_data = np.genfromtxt('output.csv').reshape([-1, 1])
# letor_input_data = np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
# letor_output_data = np.genfromtxt('datafiles/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])
# print (syn_input_data.shape)
# print (syn_output_data.shape)
print("Performing Kmeans clustering")
centroids,_ = kmeans(syn_input_data,3)
print(centroids)
idx,_ = vq(syn_input_data,centroids)
print(idx.shape)
plot(syn_input_data[idx==0,0],syn_input_data[idx==0,1],'ob',
     syn_input_data[idx==1,0],syn_input_data[idx==1,1],'or',
     syn_input_data[idx==2,0],syn_input_data[idx==2,1],'oc',)
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()
print (centroids.shape)