from .base_feature import Feature
import numpy as np
import pandas as pd
from os.path import join as join_path
import sys
sys.path.append("..")
from data import compressed_pickle as cpick

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
        afile = join_path(DATA_PATH, '{}track_uri2context.pkl.bz2'.format(subset))
        self.turi2context = cpick.load(afile)
        if logging:
            print ('CONTEXT FEATURE LOADING FINISHED')

    def transform(self, turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''
        # Get song and artist from turis
        scores = [self.turi2context[turi] for turi in turis]
        for indx, score in enumerate(scores):
            # data imputation
            if score is None:
                scores[indx] = np.ones(15)
        scores = np.array(scores)
        # print ('SCORES SHAPE:', scores.shape)
        # print ('SCORE LEN', len(scores))
        return scores
