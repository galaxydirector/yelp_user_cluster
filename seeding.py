import numpy as np
import random

def kmean_plus_plus(data_set,k):
	'''
	data_set: m x d numpy array, m is the number of users, d is the number of features
	k: number of centers
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
		picked_index = np.random.choice(users_index,p=pro)
		new_center = data_set[picked_index,:]
		centers.append(new_center)
	return centers
