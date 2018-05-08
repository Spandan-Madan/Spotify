from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
import sys
import os
from os.path import join as path_join
sys.path.append('../src')
sys.path.append('../src/data/')
sys.path.append('../src/models/')
sys.path.append('../src/features/')
sys.path.append('../src/visualization/')
from data import compressed_pickle as cpick

DATA_PATH = '../data/interim'
SAVE_PATH = '../data/sense2vec'

label = 'trackgenre'


tints2turis = cpick.load(path_join(DATA_PATH, 'track_int2track_uri.pkl.bz2'))
turis2gtopic = cpick.load(path_join(DATA_PATH, 'track_uri2gtopics.pkl.bz2'))
pl2tints = cpick.load(path_join(DATA_PATH, 'playlist2track_ints.pkl.bz2'))
print('Data loaded')
DIMENSION = 128
MIN_COUNT = 1
WORKER_NUM = 8
output_file = path_join(
    DATA_PATH, '{}_{}_{}cut'.format(label, DIMENSION,MIN_COUNT))


pl2sense = []
for pl in pl2tints:
    sentence = []
    for tint in pl:
        turi = tints2turis[tint]
        word = '{}|{}'.format(turi, turis2gtopic[turi])
        sentence.append(word)
    pl2sense.append(sentence)

#playlist = inputfile.readlines()

print ('RESPLITTING...')
#playlist = [item[:-1].split(' ') for item in playlist]
print ("STARTED TRAINING...")
model = Word2Vec(pl2sense, min_count=MIN_COUNT, size=DIMENSION,
                 workers=WORKER_NUM, negative=5, window=5)
print ("FINISHED TRAINING...")

model.save(output_file)
print ("FINISHED SAVING...")
