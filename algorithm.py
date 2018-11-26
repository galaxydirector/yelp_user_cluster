import numpy as np
import random
import time

def kmc2(data_set, k, temperature):
	n,d = data_set.shape
	centers = np.reshape(data_set[random.randint(0,n-1),:],(1,d))
	dis = distance_sq(data_set,centers)
	q = 1/2/n + 1/2*dis/np.sum(dis)
	scaled_q = np.log(q) / temperature
    scaled_q = np.exp(scaled_q - np.logaddexp.reduce(scaled_q))





	# block chain length m = 100
	m = 100
	for i in range(1,k):
		x_index = random.randint(0,n-1)
		print(i)
		for j in range(m):
			#print('m')
			y_index = np.random.choice(np.arange(n),p=scaled_q)
			if(i==1):
				d_y_sq = dis[y_index]
				d_x_sq = dis[x_index]
			else:
				d_y_sq = np.min(np.sum(np.square((data_set[y_index,:]-centers)),1))
				d_x_sq = np.min(np.sum(np.square((data_set[x_index,:]-centers)),1))
			q_x = q[x_index]
			q_y = q[y_index]
			trans_pro = np.minimum(1,d_y_sq/d_x_sq*q_x/q_y)
			x_index = np.random.choice([x_index,y_index],p=[1-trans_pro, trans_pro])
		centers = np.concatenate([centers,np.reshape(data_set[x_index,:],(1,d))],axis=0)
	return centers

def distance_sq(data,center):
	'''
	data: m x d numpy array
	center: 1D array
	return: the square of the distance between these data and the center
	'''
	distance_sq = np.sum(np.multiply((data - center),(data - center)),1)
	return distance_sq

def init_centers(data_set, k):
	""" Renzhi Version
	This algorithm serves the purpose of initialize centers by using markov chain 
	to find the furthest points to choose
	
	data_set: m x d numpy array, m is the number of users, d is the number of features
	k: number of centers
	return: the initial centers, k x d numpy array
	"""
	centers = []
	center1 = data_set[random.randint(0,len(data_set)-1)]
	centers.append(center1)
	min_dis = np.sum(np.square(np.subtract(data_set,center1)),axis=1)

	for i in range(1,k):
		pro = min_dis/np.sum(min_dis)
		picked_index = np.random.choice(np.arange(len(data_set)),p=pro)
		new_center = data_set[picked_index]
		centers.append(new_center)
		curr_dis = np.sum(np.square(np.subtract(data_set,new_center)),axis=1)
		min_dis = np.minimum(min_dis,curr_dis)
	return np.stack(centers,axis=0)

# def kmean_plus_plus(data_set,k):
# 	'''
# 	data_set: m x d numpy array, m is the number of users, d is the number of features
# 	k: number of centers
# 	return: the initial centers, k x d numpy array
# 	'''
# 	centers = []
# 	m,d = data_set.shape
# 	users_index = [i for i in range(m)]
# 	center1 = data_set[random.randint(0,m-1),:]
# 	centers.append(center1)
# 	min_dis = distance_sq(data_set,center1)
# 	for i in range(1,k):
# 		pro = min_dis/np.sum(min_dis)
# 		picked_index = np.random.choice(users_index,p=pro)
# 		new_center = data_set[picked_index,:]
# 		centers.append(new_center)
# 		curr_dis = distance_sq(data_set,new_center)
# 		min_dis = np.minimum(min_dis,curr_dis)
# 		#print(i)
# 	return np.stack(centers,axis=0)



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
		clusters[updated_index] = i ########################
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
		
		# if(i%99==0):

		centers = new_centers

	mean_distance, max_distance, min_distance = evaluate(data_set,centers)
	print("The mean distance is " + str(round(mean_distance,6)))
	print("The max distance is " + str(round(max_distance,6)))
	print("The min distance is " + str(round(min_distance,6)))
	# diff = np.sum(np.square(centers - new_centers))
	# print("The center difference is " + str(round(diff,6)))

	return centers, mean_distance, max_distance, min_distance

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
	print("shape of min_dis", min_dis.shape)
	mean_distance = np.mean(np.sqrt(min_dis))
	max_distance = np.max(np.sqrt(min_dis))
	min_distance = np.min(np.sqrt(min_dis))
	return mean_distance, max_distance, min_distance


def kmeans_singlebatch_update_yanci(batch_data, centers, eta, train = True):
	# batch have to be quite large in order to have properly update
	cluster = {key: [] for key in range(len(centers))}
	loss = 0

	if train:
		# calculate distance from point to each centers in original coordinates
		subtraction_to_centers = [np.subtract(batch_data,i) for i in centers]
	# calculate euclidean by the minibatch data set to each centers
	euclidean_distance_to_centers = np.array([np.sqrt(np.sum(np.square(np.subtract(batch_data,i)),axis=1)) for i in centers]).T

	for j in range(len(batch_data)):
		the_center,min_distance = min(enumerate(euclidean_distance_to_centers[j]), key = lambda x: x[1])
		loss += min_distance
		if train:
			cluster[the_center].append(subtraction_to_centers[the_center][j])

	print("Mean Batch Training loss is {}".format(loss/len(batch_data)))

	if train:
		gradient = [np.mean(val,axis=0) if len(val)!=0 else np.zeros(centers.shape[1]) for val in cluster.values()]
		return centers + eta*np.array(gradient)
	else:
		mean_loss = loss/len(batch_data)
		return mean_loss

def kmeans_2_yanci(data_set, k, mini_size, iteration):
	eta = 1/np.sqrt(iteration) # default learning rate
	# initialize k centers. Method 1
	# boundary_max = np.max(data_set,axis=0) 
	# boundary_min = np.min(data_set,axis=0)
	# centers = np.multiply(np.random.rand(k,len(boundary_max)),np.subtract(boundary_max,boundary_min)) + boundary_min

	# initialize k centers by randomly select k points from data set. Method 2
	centers = data_set[np.random.random_integers(0,len(data_set)-1,k)]

	data_size = len(data_set)
	# for loop iteration
	for i in range(iteration):
		# select a small dataset
		selected_ind = np.random.random_integers(0,data_size-1,mini_size)
		# centers = kmeans_minibatch_np_implementation(selected_ind, centers)
		centers = kmeans_singlebatch_update(data_set[selected_ind], centers, eta)

		if(i%99==0):
			mean_loss = kmeans_singlebatch_update(data_set, centers, eta, train = False)
			print("Mean Euclidean Distance for Test Data", mean_loss)

	return centers


def plot_cluster_generate(batch_data, centers):
	"""This function is the derivative of kmeans_singlebatch_update, 
	which serves the purpose of divide batch_data into trained k centers
	output saved in a dictionary"""

	# batch have to be quite large in order to have properly update
	cluster = {key: [] for key in range(len(centers))}

	# calculate euclidean by the minibatch data set to each centers
	euclidean_distance_to_centers = np.array([np.sqrt(np.sum(np.square(np.subtract(batch_data,i)),axis=1)) for i in centers]).T

	for j in range(len(batch_data)):
		the_center,min_distance = min(enumerate(euclidean_distance_to_centers[j]), key = lambda x: x[1])
		cluster[the_center].append(batch_data[j])

	return cluster



def main():
	path = '/Users/renzhihuang/Desktop/yelp_user_cluster/yelp.csv'
	data_set = data_import(path)
	data_set = data_normalization(data_set)
	print('Data import completed!')
	k = 50
	mini_size = 100000
	iteration = 1000
	
	s = time.time()
	centers = kmean_plus_plus(data_set,k)
	#distance_sq(data_set,centers[0,:])
	print(time.time()-s)

if __name__ == '__main__':
	main()
