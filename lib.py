from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

def design_matrix(X):
	print("Performing Kmeans clustering")
	centroids,_ = kmeans(X,3)
	print(centroids)
	idx,_ = vq(X,centroids)
	print(idx.shape)
	plot(X[idx==0,0],X[idx==0,1],'ob',
	     X[idx==1,0],X[idx==1,1],'or',
	     X[idx==2,0],X[idx==2,1],'oc',)
	plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
	show()
	print (centroids.shape)
