import gensim
import sys
import pickle
ROOT_NAME = '../data/w2v/'
FILE_NAME = 'artist_w2v_model_1min_100dim'

class Track2vec_feature(Feature):

    def __init__(self, subset=''):
        Feature.__init__(self)
        self.model = gensim.models.Word2Vec.load(ROOT_NAME_W2V + FILE_NAME)

    def transform(self,turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''
        if turis is None:
            raise ValueError('turis list is None.')

        return val = [self.model[item] for item in turis]

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
            line = [self.model[item].similarity(item, tgt) for tgt in pool]
            output.append(line)
        
        return output