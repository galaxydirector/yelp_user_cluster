import os
import time
import pandas as pd
from data_import import *
from seeding import *
# from algorithm import *

def main():
	
	path = '/Users/renzhihuang/Desktop/yelp_user_cluster/yelp.csv'
	path = os.path.expanduser('/home/aitrading/Desktop/google_drive/Course_Work/ESE545/Project3/yelp.csv')
	data_set = data_import(path)
	data_set = data_normalization(data_set)
	print('Data import completed!')
	#print(data_set)
	#print(data_set.shape) # (1326100, 17)
	#print(data_set.dtype) # float64

	#data_set = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
	#centers = kmean_plus_plus(data_set,2)
	#print(centers)
	#print(type(centers))

	k = 5
	mini_size = 100000
	iteration = 200

	centers = random_centers(data_set,k)
	# centers = kmean_plus_plus(data_set,k)
	print('Centers initialization completed!')
	#print(centers)

	centers = kmeans_2(data_set, k, mini_size, iteration,centers)
	#print(centers)

def main_2():
	path = os.path.expanduser('/home/aitrading/Desktop/google_drive/Course_Work/ESE545/Project3/yelp.csv')
	data_set = data_import(path)
	data_set = data_normalization(data_set)
	print('Data import completed!')

	#data_set = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
	#centers = kmean_plus_plus(data_set,2)
	#print(centers)
	#print(type(centers))

	k = 5
	mini_size = 100000
	iteration = 100

	kmeans_2(data_set, k, mini_size, iteration)

	# #centers = random_centers(data_set,k)
	# centers = kmean_plus_plus(data_set,k)
	# print('Centers initialization completed!')
	# #print(centers)

	# centers = kmeans_2(data_set, k, mini_size, iteration,centers)

def write_loss(data):
	"""loss writer into csv"""
	out_path = "./loss.csv"
	output = pd.DataFrame(data, columns = ['k','mean_loss'])
	output.to_csv(out_path,index=False,header=['k','mean_loss'],mode='w')

if __name__ == '__main__':
	start = time.time()
	main()
	print("duration",time.time()-start)