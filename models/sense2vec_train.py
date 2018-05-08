import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sense2vec.vectors import VectorMap
from gensim.models import Word2Vec
import sys
from os.path import join as path_join
sys.path.append('../src')
sys.path.append('../src/data/')
sys.path.append('../src/models/')
sys.path.append('../src/features/')
sys.path.append('../src/visualization/')
# will reload any library

SAVE_PATH = '../data/sense2vec'

label = 'trackgenre_128_1cut'
model_path = path_join(SAVE_PATH, label)
gensim_model = Word2Vec.load(model_path)
vector_map = VectorMap(128)

min_count = gensim_model.min_count
for string in gensim_model.wv.vocab:
    vocab = gensim_model.wv.vocab[string]
    freq, idx = vocab.count, vocab.index
    if freq < min_count:
        continue
    vector = gensim_model.wv.syn0[idx]
    vector_map.borrow(string, freq, vector)

vector_map.save(SAVE_PATH)

