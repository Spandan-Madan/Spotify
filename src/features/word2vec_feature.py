from .base_feature import Feature
import gensim
import sys
import pickle
from data import compressed_pickle as cpick
from base_feature import Feature
import numpy as np
import pandas as pd
from os.path import join as join_path
import sys
sys.path.append("..")

DATA_PATH = '../data/interim'
ROOT_NAME = '../data/w2v/'
FILE_NAME_ARTIST = 'artist_128_1cut'
FILE_NAME_ALBUM = 'album_128_1cut'
FILE_NAME_TRACK = 'track_128_1cut'


class Word2vecFeature(Feature):

    def __init__(self, subset='', ROOT_PATH=ROOT_NAME, w2v_type='artist', logging=True):
        '''
        PATH: PATH to your gensim generated word embedding. Default is artist path.
        '''
        Feature.__init__(self)

        if w2v_type == 'artist':
            PATH = ROOT_PATH + FILE_NAME_ARTIST
            afile = join_path(
                DATA_PATH, 'track_uri2artist_uri.pkl.bz2'.format(subset))
            self.mapper = cpick.load(afile)
        elif w2v_type == 'album':
            PATH = ROOT_PATH + FILE_NAME_ALBUM
            afile = join_path(
                DATA_PATH, 'track_uri2album_uri.pkl.bz2'.format(subset))
            self.mapper = cpick.load(afile)
        else:
            PATH = ROOT_PATH + FILE_NAME_TRACK

        if logging:
            print(PATH, 'IS LOADING')

        self.w2v_type = w2v_type
        self.model = gensim.models.Word2Vec.load(PATH)
        self.wv = self.model.wv
        self.wv_list = self.wv.vocab

        if logging:
            print("LOADED W2V")

    def transform(self, turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''
        if turis is None:
            raise ValueError('turis list is None.')

        idx = None
        res = []
        for turi in turis:
            if self.w2v_type == 'track':
                idx = turi
            else:
                idx = self.mapper[turi]

            res.append(idx)

        return np.array([self.model[turi] for turi in res])



class TrackFeature(Word2vecFeature):

    def __init__(self, subset='', ROOT_PATH=ROOT_NAME, logging=True):
        Word2vecFeature.__init__(self, subset, ROOT_PATH, 'track', logging)


class ArtistFeature(Word2vecFeature):

    def __init__(self, subset='', ROOT_PATH=ROOT_NAME, logging=True):
        Word2vecFeature.__init__(self, subset, ROOT_PATH, 'artist', logging)


class AlbumFeature(Word2vecFeature):

    def __init__(self, subset='', ROOT_PATH=ROOT_NAME, logging=True):
        Word2vecFeature.__init__(self, subset, ROOT_PATH, 'album', logging)
