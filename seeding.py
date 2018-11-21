def kmean_plus_plus(data_set,k):
	'''
	data_set: m x d numpy array, m is the number of users, d is the number of features
	k: number of centers
	'''
	centers = []
	m,d = data_set.shape
	data_set = data_set.tolist()
	center1 = data_set[random.randin(0,m-1)]
	centers.append(center1)
	for i in range(1,k):
		next_dis = np.zeros((m,))
		for j in range(0,i):
			curr_center = centers[j]
			curr_dis = np.sum(np.multiply((data_set-curr_center),(data_set-curr_center)),1)
			next_dis = np.minimum(curr_dis,next_dis)
		pro = next_dis/np.sum(next_dis)
		new_center = np.random.choice(data_set,pro)
		centers.append(new_center)
	return centers