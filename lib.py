import numpy as np
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

#################################
#function for calculating Kmeans clustering
#function KMeans(X, K)
#input : data set, and number of clusters required
#output : centroids calculated, spreads is the K covariance matrix
def kMeans(X, K):
	[N,D]=X.shape
	print("Performing Kmeans clustering")
	centroids,_ = kmeans(X,K)
	# print ("Printing the centroids found:")
	# print(centroids)
	idx,_ = vq(X,centroids)
	spreads=np.zeros((K, D, D))
	for k in range(0,K):
		mat = X[np.where(idx==k)[0]]
		spreads[k,:,:] = np.cov(mat.T)
		# print (spreads[k,:,:])

	# print(idx.shape)
	# plot(X[idx==0,0],X[idx==0,1],'ob',
	#      X[idx==1,0],X[idx==1,1],'or',
	#      X[idx==2,0],X[idx==2,1],'oc',)
	# plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
	# show()
	# print (centroids.shape)
	return centroids, spreads

#################################
#function for calculating Kmeans clustering
#function design_matrix(X, mu, spreads)
#input : data set, and number of clusters found, spreads
#output : design matrix using Gaussian basis function
def design_matrix(X, mu, spreads):
	# [N,D]=X.shape
	# [M,d] = mu.shape
	# phi=np.zeros((N,M));
	# for i in range(0,M):
	# 	Mpx=math.exp(-1*((X-mu[i,:]).dot(np.linalg.inv(spreads[i,:,:])).dot(np.transpose(X[i,:]-mu[i,:])))/2)
	# # print((x-mu).dot(np.linalg.inv(covarianceMat)).dot(np.transpose(x-mu)))
	# 	Sum=math.log(Mpx)+Sum
	basis_func_outputs = np.exp(np.sum(np.matmul(X - mu, spreads) * (X - mu),axis=2) / (-2)).T
	return np.insert(basis_func_outputs, 0, 1, axis=1)
