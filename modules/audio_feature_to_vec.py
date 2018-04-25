import numpy as np
import csv
import sys
import pickle
from sklearn import preprocessing

ROOT = './'
FILE_NAME = '5k_track_audiofeatures.csv'
uri2audio_feature = dict()

def inconsistant_row_num(audio_features):
	#MAKE SURE EACH LINE IS 20 LENTH LONG
	for row in audio_features:
		if len(row) != 20:
			raise ValueError('each row size is not consistant')
			
with open(ROOT + FILE_NAME) as csvfile:
	audio_features = csv.reader(csvfile, delimiter=',', quotechar='|')

	cnt = 0
	audio_feature_list = []
	for row in audio_features:
		cnt += 1
		if cnt == 10:
			break

		if len(row) != 20:
			raise ValueError('each row size is not consistant')

		row = np.array(row)
		uri = row[5]
		index = [1, 5, 14, 15, 16, 18]
		row = np.delete(row, index)
		row = np.insert(row, 0, uri, axis=0)
		audio_feature_list.append(row)

	audio_feature_list = np.array(audio_feature_list)
	uri = np.take(a=audio_feature_list, indices = [0], axis=1)
	uri = np.delete(uri, 0, axis=0)
	aud_feature_vec = np.delete(audio_feature_list, 0, axis=1)
	aud_feature_vec = np.delete(aud_feature_vec, 0, axis=0)
	aud_feature_vec_normalized = preprocessing.normalize(aud_feature_vec, norm='l2')
	aud_feature_vec_scaled = preprocessing.scale(aud_feature_vec)

	pickle.dump( uri, open( "uri.p", "wb" ) )
	pickle.dump( aud_feature_vec_normalized, open( "aud_feature_vec_normalized.p", "wb" ) )
	pickle.dump( aud_feature_vec_scaled, open( "aud_feature_vec_scaled.p", "wb" ) )

print('FINISHED')