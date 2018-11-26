# k-means algorithm
import numpy as np

def kmeans_singlebatch_update(batch_data, centers, eta, train = True):
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

def kmeans_2(data_set, k, mini_size, iteration):
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

def kmean_plus_plus(data_set,k):
	'''
	data_set: m x d numpy array, m is the number of users, d is the number of features
	k: number of centers
	return: the initial centers, k x d numpy array
	'''
	eta = 1/np.sqrt(iteration) # default learning rate
	# initialize k centers


	data_size = len(data_set)
	# for loop iteration
	for i in range(iteration):
		# select a small dataset
		selected_ind = np.random.random_integers(0,data_size-1,mini_size)
		# centers = kmeans_minibatch_np_implementation(selected_ind, centers)
		centers = kmeans_singlebatch_update(data_set[selected_ind], centers, eta)

		if(i%10==0):
			mean_loss = kmeans_singlebatch_update(data_set, centers, eta, train = False)
			print("Mean Euclidean Distance for Test Data", mean_loss)

	return centers


if __name__ == '__main__':
	centers = kmeans_2(data_set, k, mini_size)



