from os.path import join as path_join
import numpy as np
from gensim.models import Word2Vec
DATA_PATH = '../data/pooling'

class W2VPooler(object):

    def __init__(self,verbose=True):
        afile = path_join(DATA_PATH,'model_min5_new.bin')
        self.model = Word2Vec.load(afile)
        self.verbose=verbose
        return

    def get_vecs(self,seeds):
        n = len(seeds)
        vecs = []
        found = np.zeros(n,dtype=bool)
        for indx,turi in enumerate(seeds):
            track = 'spotify:track:{}'.format(turi)
            try:
                vecs.append(self.model[track])
                found[indx]=1
            except:
                if self.verbose:
                    print(uri_to_name_artist[track],track)
        if self.verbose:
            print('Found {} out of {} vecs'.format(np.sum(found),len(seeds)))

        return np.array(vecs),found

    def rec_average(self,seeds, n=1000):
        vecs, found = self.get_vecs(seeds)

        if self.verbose:
            print('Averaging representation, returning pool of size {}'.format(n))

        representation = np.mean(np.array(vecs),axis=0)
        # find most similar
        results = self.model.most_similar([representation],topn=n)
        pool_dist = [i[1] for i in results]
        pool_turi = [i[0].replace('spotify:track:','') for i in results]
        return pool_turi, pool_dist
