from data_import import *
from seeding import *

def main():
	path = '/Users/renzhihuang/Desktop/yelp_user_cluster/yelp.csv'
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

	k = 20
	mini_size = 100000
	iteration = 1000

	#centers = random_centers(data_set,k)
	centers = kmean_plus_plus(data_set,k)
	print('Centers initialization completed!')
	#print(centers)

	centers = kmeans_2(data_set, k, mini_size, iteration,centers)
	#print(centers)

if __name__ == '__main__':
	main()