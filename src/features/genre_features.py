from base_feature import Feature
import numpy as np
import pandas as pd
import gensim
from os.path import join as join_path
import sys
sys.path.append("..")
from data import compressed_pickle as cpick
from gensim.models.wrappers import FastText
import fastText

DATA_PATH = '../data/interim'
FASTTEXT_PATH = "../data/fasttext/wiki.en.bin"

'''
INSTRUCTION:
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
(IF YOU ARE INSTALLING THIS ON A LINUX MACHINE)
$ sudo apt-get install g++
$ pip install .

YOU NEED TO DOWNLOAD FASTTEXT BIN FILE TOO FOR ENGLISH.
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip

Unzip it and use the path to this file as FASTTEXT_PATH
'''


class GenreFeatures(Feature):

    def __init__(self, subset='', fasttext_path=FASTTEXT_PATH, logging=True):
        '''
        PATH: PATH to your gensim generated word embedding. Default is artist path.
        '''
        Feature.__init__(self)
        afile = join_path(DATA_PATH, 'genres.pkl.bz2'.format(subset))
        afile_mapper = join_path(
            DATA_PATH, 'track_uri2artist_uri.pkl.bz2'.format(subset))

        if logging:
            print("LOADING GENRE..")
        self.df = cpick.load(afile)
        self.mapper = cpick.load(afile_mapper)

        if logging:
            print("LOADING FASTTEXT...")

        self.model = fastText.load_model(FASTTEXT_PATH)

        if logging:
            print("FINSIHED LOADING FASTTEXT..")
            print("FINISHED LOADING GENRE FEATURE")

    def transform(self, turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''
        if turis is None:
            vals = np.array([v for v in list(self.df.values())])

        output = []
        for turi in turis:
            artist_uri = self.mapper[turi]
            row = self.df[artist_uri]
            if len(row) == 0:
                artist_vec = np.array([0.0] * 300)
                output.append(artist_vec)
                continue
            artist_vec = np.array(
                [self.model.get_sentence_vector(genre) for genre in row])
            artist_vec = np.mean(artist_vec, axis=0)
            output.append(artist_vec)
        x = np.array(output)
        return x


class GenreLDA(Feature):

    def __init__(self, subset='', logging=True):
        '''

        '''
        Feature.__init__(self)
        afile = join_path(DATA_PATH, 'track_uri2gtopics_vec.pkl.bz2'.format(subset))
        self.turi2gvec = cpick.load(afile)

    def transform(self, turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''
        x = np.array([self.turi2gvec[turi] for turi in turis])
        return x
