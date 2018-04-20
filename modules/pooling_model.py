
# coding: utf-8

# # Neighbourhood Finder demo

# # Imports and helper functions

# In[ ]:


import json
from random import randint
import os
import json
import lyricwikia
import pickle
import sys
import numpy as np

def artists_in_playlist(playlist):
    all_ = []
    for track in playlist['tracks']:
        all_.append(track['artist_name'])
    return list(set(all_))

def albums_in_playlist(playlist):
    all_ = []
    for track in playlist['tracks']:
        all_.append(track['album_name'])
    return list(set(all_))

def tracks_in_playlist(playlist):
    all_ = []
    for track in playlist['tracks']:
        all_.append(track['track_uri'])
    return list(set(all_))

def track_names_in_playlist(playlist):
    all_ = []
    for track in playlist['tracks']:
        all_.append(track['track_name'])
    return list(set(all_))

def lyrics_in_playlist(playlist):
    all_ = {}
    info = {}
    ct = 0
    failed = []
    for track in playlist['tracks']:
        track_uri = track['track_uri']
        song_name = track['track_name']
        artist_name = track['artist_name']
        try:
            lyrics = lyricwikia.get_lyrics(artist_name, song_name)
            all_[track_uri] = lyrics
            info[track_uri] = (song_name,artist_name)
        except:
            failed.append(ct)
        ct += 1
    return all_,failed,info

    

# # Pick song name by looking up for artist

# In[8]:




# In[10]:



# INFILE_PATH = sys.argv[1]
# f = open(INFILE_PATH,'r')
# content = f.readlines()

# list_s = []
# for c in content:
#     song,artist = c.rstrip().split('$$$$')
#     list_s.append((song,artist))

# list_s = [('The Scientist','Coldplay'),('Immigrant Song','Led Zeppelin'),('T.N.T.','AC/DC')]
def pooled_songs(list_s,VERBOSE,pickles_folder):
    print('Loading pickle files, please wait!')
    f = open(pickles_folder+'uri_to_track_info.pckl','rb')
    uri_to_name_artist = pickle.load(f)
    f.close()

    f = open(pickles_folder+'artist_to_songs.p','rb')
    artist_to_songs = pickle.load(f)
    f.close()

    clean_artist_to_songs = {}
    for artist in artist_to_songs.keys():
        clean_artist_to_songs[artist] = list(set(artist_to_songs[artist]))

    name_artist_to_uri = {}
    for uri in uri_to_name_artist.keys():
        tup = uri_to_name_artist[uri]
        name_artist_to_uri[tup] = uri

    from gensim.models import Word2Vec
    model = Word2Vec.load(pickles_folder+'/model_min5_new.bin')

    list_already = []
    song_names = []
    artist_names = []
    for l in list_s:
        song_names.append(l[0])
        artist_names.append(l[1])
        uri = name_artist_to_uri[l].split(':')[-1]
        list_already.append(uri)
    sample = np.zeros(300)
    ct = 0
    for song in list_already:
        
        track = 'spotify:track:%s'%song
        try:
            ct += 1
            sample += model[track]
        except:
            if VERBOSE == 'True':
                print(uri_to_name_artist[track],track)
    representation = sample/ct

    most_sim = model.most_similar([representation],topn=1000)
    artists = []
    songs = []
    for suggestion in most_sim:
        
        songs.append(uri_to_name_artist[suggestion[0]])
        artists.append(uri_to_name_artist[suggestion[0]][1])
    #     print(uri_to_name_artist[suggestion[0]])

    from collections import Counter
    c = Counter(artists)
    top_artists = c.most_common(10)

    top_songs_to_report = songs[:25]
    if VERBOSE == 'True':
        print('Top songs - ')
        print(top_songs_to_report,sep='\n')
        print('\n\n\nTop Artisis')

    for t in top_artists:
        top_artists_to_report = t[0]

    if VERBOSE == 'True':
        print(top_artists_to_report)

    most_sim = model.most_similar([representation],topn=1000)
    artists = []
    songs = []
    for suggestion in most_sim:
        
        songs.append(uri_to_name_artist[suggestion[0]])
        artists.append(uri_to_name_artist[suggestion[0]][1])
    #     print(uri_to_name_artist[suggestion[0]])

    from collections import Counter
    c = Counter(artists)
    top_artists = c.most_common(14)

    
    printed = 0
    i = 0
    reported_ = []
    while(printed<25):
        song,artist = songs[i]
        if song not in song_names:
            if artist not in artist_names:
                reported_.append((song,artist))
                if VERBOSE == 'True':
                    print('Top songs - ')
                    print(song,artist)
                printed += 1
        i += 1
    
    non_repeated_artists = []
    
    for t in top_artists:
        if t not in artist_names:
            non_repeated_artists.append(t[0])
            if VERBOSE == 'True':
                print('\n\n\nTop Artisis')
                print(t[0])

    return reported_,non_repeated_artists


