import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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