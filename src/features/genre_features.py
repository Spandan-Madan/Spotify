from base_feature import Feature
import numpy as np
import pandas as pd
import gensim
import sys
from os.path import join as join_path
sys.path.append("..")
from data import compressed_pickle as cpick
from gensim.models.wrappers import FastText
import fastText
from scipy import spatial


DATA_PATH = '../../data/genre'
FASTTEXT_PATH = "/Users/timlee/Downloads/wiki.en/wiki.en.bin"

'''
INSTRUCTION:
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .

YOU NEED TO DOWNLOAD FASTTEXT BIN FILE TOO FOR ENGLISH.
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip

Unzip it and use the path to this file as FASTTEXT_PATH
'''
class genre_feature(Feature):

	def __init__(self, subset='', fasttext_path = FASTTEXT_PATH):
		'''
		PATH: PATH to your gensim generated word embedding. Default is artist path.
		'''
		Feature.__init__(self)
		afile = join_path(DATA_PATH, '{}genres.pkl.bz2'.format(subset))
		self.df = cpick.load(afile)
		print ("LOADING FASTTEXT...")
		self.model = fastText.load_model(FASTTEXT_PATH)
		print ("FINSIHED LOADING..")

	def transform(self,turis):
		'''
		Convert track a list of turis to a vector representation.
		Input: turis is a list of strings of length n
		Output: will return a vector of shape (n_dim,n)
		'''
		if turis is None:
			raise ValueError('turis list is None.')

		output = []
		for item in turis:
			row = self.df[item]
			artist_vec = np.array([self.model.get_sentence_vector(genre) for genre in row])
			artist_vec = np.mean(artist_vec, axis=0)
			output.append(artist_vec)
		return output

	def distance(self,seeds,pool):
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

a = genre_feature()
uri = ['2wIVse2owClT7go1WT98tk', '6vWDO969PvNqNYHIOW5v0m']
a.transform(uri)