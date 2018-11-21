import pandas as pd
import numpy as np

def data_import(path):
	raw_data = pd.read_csv(path)
	feature_list = []
	user_id = raw_data['user_id']
	user_name = raw_data['name']
	for key in raw_data.keys():
		if(key=='user_id' or key=='name'):
			continue
		elif(key=='yelping_since'):
			continue
		elif(key=='elite'):
			continue
		else:
			feature_list.append(raw_data[key].values)
	data_set = np.stack(feature_list,axis=1)
	return data_set

path = '/Users/renzhihuang/Desktop/yelp_user_cluster/yelp.csv'
data_set = data_import(path)
print(data_set.shape)
print(data_set.dtype)

