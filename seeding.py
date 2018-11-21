import numpy as np
import random

def distance_sq(data,center):
	'''
	data: m x d numpy array
	center: 1D array
	return: the square of the distance between these data and the center
	'''
	distance_sq = np.sum(np.multiply((data - center),(data - center)),1)
	return distance_sq

def kmean_plus_plus(data_set,k):
	'''
	data_set: m x d numpy array, m is the number of users, d is the number of features
	k: number of centers
	return: the initial centers, k x d numpy array
	'''
	centers = []
	m,d = data_set.shape
	users_index = [i for i in range(m)]
	center1 = data_set[random.randint(0,m-1),:]
	centers.append(center1)
	for i in range(1,k):
		next_dis = np.sum(np.multiply((data_set-centers[0]),(data_set-centers[0])),1)
		for j in range(1,i):
			curr_center = centers[j]
			curr_dis = np.sum(np.multiply((data_set-curr_center),(data_set-curr_center)),1)
			next_dis = np.minimum(curr_dis,next_dis)
		pro = next_dis/np.sum(next_dis)
		#print(pro)
		picked_index = np.random.choice(users_index,p=pro)
		new_center = data_set[picked_index,:]
		centers.append(new_center)
	return np.stack(centers,axis=0)

def random_centers(data_set,k):
	'''
	data_set: m x d numpy array, m is the number of users, d is the number of features
	k: number of centers
	return: the initial centers, k x d numpy array
	'''
	centers = []
	m,d = data_set.shape
	centers = data_set[np.random.random_integers(0,m-1,k),:]
	return centers

def find_cluster(data_set,centers):
	'''
	data_set: m x d numpy array, m is the number of users, d is the number of features
	centers: the center points
	return: m x 1 array indicating the cluster for each points
	'''
	k,d = centers.shape
	m,d = data_set.shape
	clusters = np.zeros(m)
	min_dis = distance_sq(data_set,centers[0,:])
	for i in range(1,k):
		curr_dis = distance_sq(data_set,centers[i,:])
		updated_index = np.where(curr_dis<min_dis)[0]
		clusters[updated_index] = i
		min_dis[updated_index] = curr_dis[updated_index]
	return clusters


def kmeans_singlebatch_update(batch_data, centers, eta):
	'''
	batch_data: m x d numpy array
	centers: k x d numpy array
	eta: learning rate
	'''
	k,d = centers.shape
	m,d = batch_data.shape
	clusters = find_cluster(batch_data,centers)
	new_centers = np.zeros((k,d))
	for i in range(0,k):
		curr_cluster = batch_data[np.where(clusters==i)[0],:]
		if(curr_cluster.shape[0]==0):
			continue
		else:
			#print(centers[i,:].shape)
			#print(np.mean((curr_cluster - centers[i,:]),axis = 0).shape)
			#print(curr_cluster)
			new_centers[i,:] = centers[i,:] + eta * np.mean((curr_cluster - centers[i,:]),axis = 0)
	return new_centers


def kmeans_2(data_set, k, mini_size, iteration,centers):
	# for loop iteration
	m,d = data_set.shape
	for i in range(1,iteration):
		# select a small dataset
		selected_ind = np.random.random_integers(0,m-1,mini_size)
		new_centers = kmeans_singlebatch_update(data_set[selected_ind,:], centers, 1/i)
		
		if(i%50==0):
			mean_distance = evaluate(data_set,new_centers)
			print("The mean distance is " + str(round(mean_distance,6)))
			diff = np.sum(np.square(centers - new_centers))
			print("The center difference is " + str(round(diff,6)))
		centers = new_centers
	return centers

def evaluate(data_set,centers):
	'''
	given the final centers, return the mean square distance for the dataset
	'''
	k,d = centers.shape
	m,d = data_set.shape
	clusters = np.zeros(m)
	min_dis = distance_sq(data_set,centers[0,:])
	for i in range(1,k):
		curr_dis = distance_sq(data_set,centers[i,:])
		updated_index = np.where(curr_dis<min_dis)[0]
		clusters[updated_index] = i
		min_dis[updated_index] = curr_dis[updated_index]
	mean_distance = np.mean(min_dis)
	return mean_distance
