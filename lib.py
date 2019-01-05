import numpy as np
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans2,vq

#################################
#function for calculating Kmeans clustering
#function KMeans(X, K)
#input : data set, and number of clusters required
#output : centroids calculated, spreads is the K covariance matrix
def kMeans(X, K):
	# np.random.shuffle(X)
	[N,D]=X.shape
	Iterater=10
	# print("Performing Kmeans clustering: ", K)
	centroids=np.zeros((K, D))
	for i in range(1,Iterater+1): #running times
		centrds,_ = kmeans2(X,K,minit='points')
		centroids=centroids+centrds
	centroids=centroids/Iterater

	# print ("Printing the centroids found:")
	# print(centroids)np.linalg.inv
	idx,_ = vq(X,centroids)
	spreads=np.zeros((K, D, D))
	for k in range(0,K):
		mat = X[np.where(idx==k)[0],:]
		#print (mat.shape)
		spreads[k,:,:] = np.linalg.pinv(np.cov(mat.T))#think about this line
		# spreads[k,:,:] = np.cov(mat.T)
		# print (spreads[k,:,:].shape)

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
	# [N,D]=X.shapeg
	# [M,d] = mu.shape
	# phi=np.zeros((N,M));
	# for i in range(0,M):
	# 	Mpx=math.exp(-1*((X-mu[i,:]).dot(np.linalg.inv(spreads[i,:,:])).dot(np.transpose(X[i,:]-mu[i,:])))/2)
	# # print((x-mu).dot(np.linalg.inv(covarianceMat)).dot(np.transpose(x-mu)))
	# 	Sum=math.log(Mpx)+Sum
	basis_func_outputs = np.exp(np.sum(np.matmul(X - mu, spreads) * (X - mu),axis=2) / (-2)).T
	return np.insert(basis_func_outputs, 0, 1, axis=1)

#################################
#function for calculating the closed form solution
#function closed_form_sol(L2_lambda, design_matrix, output_data)
#input : L2_lambda, design_matrix, and output_data
#output : res
def closed_form_sol(L2_lambda, design_matrix, output_data):
	return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) +
	np.matmul(design_matrix.T, design_matrix),np.matmul(design_matrix.T,output_data)).flatten()

#################################
#function for calculating the gradient descent solution
#function sgd_solution(L2_lambda, design_matrix, output_data)
#input : learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data
#output : weights
def sgd_solution(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data):
	[N,D]=design_matrix.shape
	E=0
	weights = np.zeros([1,D])
	for epoch in range(num_epochs):
		
		for i in range(int(N/minibatch_size)):
			lower_bound = i * minibatch_size
			upper_bound = min((i+1)*minibatch_size, N)
			Phi = design_matrix[lower_bound : upper_bound, :]
			t = output_data[lower_bound : upper_bound, :]
			E_D = np.matmul((np.matmul(Phi, weights.T)-t).T, Phi)
			E = (E_D + L2_lambda * weights) / minibatch_size
			weights = weights - learning_rate * E
		# print (weights.shape)
		# print(np.linalg.norm(E))
		# print (weights)
	return weights.flatten()

#################################
#function for calculating the gradient descent solution
#function sgd_solution(L2_lambda, design_matrix, output_data)
#input : learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data
#output : weights
def sgd_solution_early_stop(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, design_matrix_val, Y_Val):
	[N,D]=design_matrix.shape
	E=0
	weights = np.zeros([1,D])
	valError = float("inf")#defining the infinite value for the validation error, initially
	weights_star = np.zeros([1,D]) #weight to store the optimum weights
	I=0
	I_star=0#actual training steps
	j=0#tracking the patience 
	P=10# patience value
	for epoch in range(num_epochs):
		for i in range(int(N/minibatch_size)):
			lower_bound = i * minibatch_size
			upper_bound = min((i+1)*minibatch_size, N)
			Phi = design_matrix[lower_bound : upper_bound, :]
			t = output_data[lower_bound : upper_bound, :]
			E_D = np.matmul((np.matmul(Phi, weights.T)-t).T, Phi)
			E = (E_D + L2_lambda * weights) / minibatch_size
			weights = weights - learning_rate * E
		# print (weights.shape)
		valE = erms(design_matrix_val, Y_Val, weights.T, L2_lambda)
		# print (epoch)
		if valE<valError:
			valError=valE
			j=0; #resetting the j
			I_star = I
			weights_star = weights
		elif j==P-1:
			print ("Early stopped at epoch: ", epoch)
			break
		else:
			j=j+1
		# print(np.linalg.norm(E))
		# print (weights)
	return weights_star.flatten()

#Error function
def erms(design_matrix, Y, W, L2_lambda):
	[N, D] = design_matrix.shape
	Y_dash = design_matrix.dot(W)
	# print(Y_dash[1:10])
	# print("min Y max Y = %0.4f  %0.4f"%(np.min(Y_dash), np.max(Y_dash)))
	Error =  np.sum(np.square(Y - Y_dash))/2 + 0.5*L2_lambda*(W.T.dot(W))
	Erms = np.sqrt((2 * Error)/N)
	return Erms[0, 0]

