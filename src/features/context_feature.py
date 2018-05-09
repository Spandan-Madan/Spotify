from .base_feature import Feature
import numpy as np
import pandas as pd
from os.path import join as join_path
import sys
sys.path.append("..")
from data import compressed_pickle as cpick
from scipy import spatial
from ast import literal_eval
MODEL_PATH = '../data/context'
# MODEL_PATH = '/Users/mehulsmritiraje/Desktop/Harvard_ME_in_CSE/Spring_2018/AC_297r/Spotify/data/context'
DATA_PATH = '../data/interim'
# DATA_PATH = '/Users/mehulsmritiraje/Desktop/Harvard_ME_in_CSE/Spring_2018/AC_297r/Spotify/data/interim'

class ContextFeatures(Feature):

    def __init__(self, subset='', logging=True):
        '''
        Feature constructor, will load all relevant data, if using the 5k subset, use the string "5k-"
        Input: can be whatever you need (dimention of embedding, folder,etc)
        '''
        Feature.__init__(self)
        if logging:
            print ('CONTEXT FEATURE LOADING')
        # file = join_path(
        #    MODEL_PATH, '{}data_words_one_hot.pkl.bz2'.format(subset))
        # self.context_model = cpick.load(file)
        self.track_data = pd.read_csv(join_path(DATA_PATH, '{}turi2context.csv'.format(subset)))
        if logging:
            print ('CONTEXT FEATURE LOADING FINISHED')

    def transform(self, turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''
        # Get song and artist from turis
        scores = []
        for uri in turis:
            uri = 'spotify:track:' + uri
            row = self.track_data[self.track_data['uri'] == uri]
            ele = row['scores'].values
            scores.append(literal_eval(ele[0]))

        scores = np.array(scores)
        # print ('SCORES SHAPE:', scores.shape)
        # print ('SCORE LEN', len(scores))
        return scores

    def distance(self, seed, pool):
        '''
        Calculate the distance between two sets of tracks (seeds) and (pools). Will return a numpy array of distances.
        Input: seeds is a list of track_uris (strings)
               pool is a list
        Output: will return a vector of shape (n_dim,n)
        '''
        seed = self.transform(seed)
        pool = self.transform(pool)

        output = []
        for src in seed:
            line = [spatial.distance.cosine(src, tgt) for tgt in pool]
            output.append(line)

        return output
