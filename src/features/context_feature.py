from .base_feature import Feature
import numpy as np
import pandas as pd
from os.path import join as join_path
import sys
sys.path.append("..")
from data import compressed_pickle as cpick
from scipy import spatial
MODEL_PATH = '../data/context'
# MODEL_PATH = '/Users/mehulsmritiraje/Desktop/Harvard_ME_in_CSE/Spring_2018/AC_297r/Spotify/data/context'
DATA_PATH = '../data/interim'
# DATA_PATH = '/Users/mehulsmritiraje/Desktop/Harvard_ME_in_CSE/Spring_2018/AC_297r/Spotify/data/interim'


class ContextFeatures(Feature):

    def __init__(self, subset=''):
        '''
        Feature constructor, will load all relevant data, if using the 5k subset, use the string "5k-"
        Input: can be whatever you need (dimention of embedding, folder,etc)
        '''
        Feature.__init__(self)
        file = join_path(
            MODEL_PATH, '{}data_words_one_hot.pkl.bz2'.format(subset))
        self.context_model = cpick.load(file)

        self.track_data = pd.read_csv(join_path(DATA_PATH, '{}5k_track_uri.csv'.format(subset)))
        return

    def transform(self, turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''
        # Get song and artist from turis
        songs_artists = []
        for uri in turis:
            row = self.track_data[self.track_data['uri'] == uri]
            ele = (row['title'].values[0], row['artist'].values[0])
            songs_artists.append(ele)

        scores = []
        for song in songs_artists:
            score = self.context_model[(self.context_model['song'] == song[0]) &\
                (self.context_model['artist'] == song[1])]['one_hot']
            for s in score:
                scores.append(s)
        return scores