import os
import time
import pandas as pd
from data_import import *
from algorithm import *
from plotting import *


def write_loss(data):
	"""loss writer into csv"""
	out_path = "./loss_mcmcmodified.csv"
	output = pd.DataFrame(data, columns = ['k','mean_loss','max_loss','min_loss'])
	output.to_csv(out_path,index=False,header=['k','mean_loss','max_loss','min_loss'],mode='w')

def batch_train():
	# path = '/Users/renzhihuang/Desktop/yelp_user_cluster/yelp.csv'
	path = os.path.expanduser('/home/aitrading/Desktop/google_drive/Course_Work/ESE545/Project3/yelp.csv')
	data_set = data_import(path)
	data_set = data_normalization(data_set)
	print('Data import completed!')
	#print(data_set)
	#print(data_set.shape) # (1326100, 17)
	#print(data_set.dtype) # float64

	start = time.time()
	writedata = []
	for k in list(range(5,100,5))+list(range(100,520,20)):
		print("current k is ", k)
		mini_size = 100000
		iteration = 50

		# centers = random_centers(data_set,k)
		# centers = kmean_plus_plus(data_set,k)
		# centers = kmc2(data_set, k, temperature = 0.8)
		print('Centers initialization completed!')

		centers, mean_distance, max_distance, min_distance = kmeans_2(data_set, k, mini_size, iteration,centers)
		writedata.append((k,mean_distance, max_distance, min_distance))
		print("finish k centers within {}s".format(time.time()-start))

	write_loss(writedata)

def plot_2D_cluster():
	path = os.path.expanduser('/home/aitrading/Desktop/google_drive/Course_Work/ESE545/Project3/yelp.csv')
	data_set = data_import(path)
	data_set = data_normalization(data_set)
	print('Data import completed!')

	k = 5
	mini_size = 100000
	iteration = 50

	# train
	centers = kmean_plus_plus(data_set,k)
	centers, mean_distance, max_distance, min_distance = kmeans_2(data_set, k, mini_size, iteration,centers)


	# plot
	output_path = os.path.expanduser('./2Dplot.png')
	dim1 = 2 # column of funny
	dim2 = 5 # column of average rates
	plot_clusters(data_set, cluster, dim1, dim2, output_path)

if __name__ == '__main__':
	start = time.time()
	plot_2D_cluster()
	print("duration",time.time()-start)