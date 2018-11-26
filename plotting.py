import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# k = [5, 10, 20, 30, 50, 100, 200, 300, 500]
# mean = [0.41, 0.33, 0.35, 0.36, 0.43, 0.451, 0.52, 0.63, 0.75]
# minimum = [0.49, 0.2, 0.37, 0.44, 0.56, 0.51, 0.62, 0.63, 0.85]
# maximum = [0.31, 0.13, 0.25, 0.26, 0.43, 0.451, 0.562, 0.613, 0.95]

# plt.title('Error based on the initialization')
# plt.plot(k, mean, color='green', label='mean distance')
# plt.plot(k, minimum, color='red', label='minimum distance')
# plt.plot(k, maximum, color='skyblue', label='maximum distance')
# plt.legend()

# plt.xlabel('k value')
# plt.ylabel('distance')
# plt.show()



def plot_clusters(data, cluster, dim1, dim2, output_path):
	"""
	data & cluster produce clustered data saved in a dict
	dim1 and dim2 are features in interest
	"""
	# clustered data is saved in a dict
	clustered_data = plot_cluster_generate(data,cluster)
	
	# generate plots
	colors = cm.rainbow(np.linspace(0, 1, len(clustered_data)))
	for cluster, c in zip(clustered_data.values(),colors):
		x = []
		y = []
		for data in cluster:
			x.append(data[dim1])
			y.append(data[dim2])

		plt.scatter(x, y, color=c)

	plt.xlabel('dimension {}'.format(dim1))
	plt.ylabel('dimension {}'.format(dim2))
	path = os.path.join(output_path, '{}.png'.format('clusters'))
	plt.savefig(path)



if __name__ == '__main__':
	output_path = os.path.expanduser('./mfucker.png')
	clustered_data = {0:[[1,2,3],[2,2,4]],1:[[4,5,6],[4,6,6]],2:[[8,9,10],[10,12,11]]}

	dim1 = 1
	dim2 = 2
	# generate plots
	colors = cm.rainbow(np.linspace(0, 1, len(clustered_data)))
	for cluster, c in zip(clustered_data.values(),colors):
		x = []
		y = []
		for data in cluster:
			x.append(data[dim1])
			y.append(data[dim2])

		plt.scatter(x, y, color=c)
		
	# path = os.path.join(output_path, '{}.pdf'.format(filename))
	# plt.savefig(output_path)
	plt.show()