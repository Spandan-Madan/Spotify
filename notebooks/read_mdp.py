from os import listdir
from os.path import isfile, join
import json
import sys
import numpy as np
import pickle


mypath = "./spotify_stuff/spotify_data/data"
onlyfiles = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
sentences = set()

total_num = len(onlyfiles)
fout = open("playlist_bin.txt", 'w')
print (total_num)
cnt = 0
for file in onlyfiles:
	cnt += 1
	if cnt * 1.0 % 100 == 0:
		print (cnt * 100.0 / total_num)
	data = json.load(open(mypath+'/'+file))
	for playlist in data['playlists']:
		fout.write(' '.join([song['track_uri']for song in playlist['tracks']]) + '\n')

fout.close()


total_num = len(onlyfiles)
print (total_num)
song_artist_lookup = dict()
cnt = 0
for file in onlyfiles:
	cnt += 1
	if cnt * 1.0 % 100 == 0:
		print (cnt * 100.0 / total_num)
	data = json.load(open(mypath+'/'+file))
	for playlist in data['playlists']:
		for song in playlist['tracks']:
			if song['track_uri'] not in song_artist_lookup:
				song_artist_lookup[song['track_uri']] = (song['track_name'], song['artist_name'])

print ("saving...")
f_save = open('look_up.p','wb')
pickle.dump(song_artist_lookup,f_save)
f_save.close()
print ("FINISHED...")