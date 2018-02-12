import os
import json
import lyricwikia
import pickle
import sys

base_path = '/Users/spandanmadan/Desktop/Spotify/mpd/data/'
all_slices = os.listdir('mpd/data/')
track_uri_to_lyrics = {}

ct = 0
slice_count = 0
tracks_failed = {}
for data_slice in all_slices:
	print("Working on slice",slice_count)
	sys.stdout.flush()
	slice_count += 1
	f = open(base_path + data_slice,'r')
	data = json.load(f)
	f.close()
	print('opened')
	sys.stdout.flush()

	for playlist in data['playlists']:
		tracks = playlist['tracks']
		for track in tracks:
			if ct == 500:
				print('Reached ',ct,' tracks')
				sys.stdout.flush()
			ct += 1
			artist = track['artist_name']
			song = track['track_name']
			tup = (artist,song)
			uri = track['track_uri']
			if uri in track_uri_to_lyrics.keys() or uri in tracks_failed.keys():
				continue
			else:
				try:
					lyrics = lyricwikia.get_lyrics(artist, song)
					track_uri_to_lyrics[uri] = (song,artist,lyrics)
				except:
					tracks_failed[uri] = (song,artist)

f_save = open('song_artist_lyrics.p','wb')
pickle.dump(track_uri_to_lyrics,f_save)
f_save.close()


f_save = open('failed_tracks.p','wb')
pickle.dump(tracks_failed,f_save)
f_save.close()