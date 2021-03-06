# k-means algorithm

import numpy as np





def kmeans_original(data, k):
	# initialize k centers
	boundary_max = max(data) 
	boundary_min = min(data)
	centers = np.random.random()*(max-min)+min # iterate for all features and all centers

	# 1. while loop iteration
	# stop criteria is the loss

	# 2. for loop iteration
	# stop criteria is either the loss or end of iteration
	# then print the loss

	# implement using for loop iteration
	for i in range(ite):
		# create a dict to save data points by cluster
		cluster = {key: [] for key in range(k)}
		loss = 0
		############ can we not use for loop but use dataframe to process it?###############
		# go through all data points
		for m in data:
			euclidean_distance_to_centers = [sqrt(sum((m-i)^2)) for i in centers]
			the_center,min_distance = min(enumerate(euclidean_distance_to_centers), key = lambda x: x[1])
			loss += min_distance
			cluster[the_center].append(m)

			# update centers
			centers = [np.mean(cluster[the_center],axes=0) for i in range(k)] # average all the rows

def kmeans_minibatch_np_implementation_original(batch_data_ind, centers):
	# batch have to be quite large in order to have properly update
	cluster = {key: [] for key in range(k)}
	loss = 0
	############ can we not use for loop but use dataframe to process it?###############
	# go through all data points
	for ind in batch_data_ind:
		m = data_set[ind]
		euclidean_distance_to_centers = [np.sqrt(np.sum(np.square(np.subtract(m,i)),axis=0)) for i in centers]
		the_center,min_distance = min(enumerate(euclidean_distance_to_centers), key = lambda x: x[1])
		loss += min_distance
		cluster[the_center].append(m)

	# update centers
	return [np.mean(cluster[the_center],axis=0) for i in range(k)]




def kmeans_singlebatch_update_old2(batch_data, centers, eta):
	# batch have to be quite large in order to have properly update
	cluster = {key: [] for key in range(k)}
	loss = 0
	############ can we not use for loop but use dataframe to process it?###############
	
	# calculate euclidean by the minibatch data set to each centers 
	euclidean_distance_to_centers = np.array([np.sqrt(np.sum(np.square(np.subtract(batch_data,i)),axis=1)) for i in centers]).T

	for j in range(len(batch_data)):
		single_row_distance = euclidean_distance_to_centers[j]
		the_center,min_distance = min(enumerate(single_row_distance), key = lambda x: x[1])
		loss += min_distance
		cluster[the_center].append(m)

	# update centers
	return [np.mean(cluster[the_center],axis=0) for i in range(k)]



if __name__ == '__main__':
	centers = kmeans_2(data_set, k, mini_size)



