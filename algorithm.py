# k-means algorithm







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











def kmeans_minibatch_pd_implementation(batch_data, centers):

def kmeans_2(data_set, k, mini_size):
	# initialize k centers
	boundary_max = np.max(data,axis=0) 
	boundary_min = np.min(data,axis=0)
	centers = np.multiply(np.random.rand(k,len(boundary_max)),np.subtract(boundary_max-boundary_min)) + boundary_min

	data_size = len(data_set)
	# for loop iteration
	for i in range(ite):
		# select a small dataset
		selected_ind = np.random.random_integers(0,data_size-1,mini_size)
		centers = kmeans_minibatch_np_implementation(selected_ind, centers)

	return centers

def kmeans_minibatch_np_implementation(batch_data_ind, centers):
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