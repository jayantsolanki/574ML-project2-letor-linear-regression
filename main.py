#main.py
import numpy as np
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from lib import *
syn_input_data = np.loadtxt('input.csv', delimiter=',')
syn_output_data = np.genfromtxt('output.csv').reshape([-1, 1])
# letor_input_data = np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
# letor_output_data = np.genfromtxt('datafiles/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])
# print (syn_input_data.shape)
# print (syn_output_data.shape)

design_matrix(syn_input_data)