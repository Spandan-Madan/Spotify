from os.path import join as path_join
import numpy as np
from gensim.models import Word2Vec
DATA_PATH = '../data/pooling'
from sklearn.cluster import KMeans
from math import ceil
from models.sklearn_KMedoids import KMedoids

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

    def recommend(self,seeds,n=1000,agg_strat='mean',**kwargs):
        if agg_strat=='mean' or len(seeds)==1:
            return self.mean_pool(seeds,n,**kwargs)
        elif agg_strat=='centroids':
            return self.centroids_pool(seeds,n,**kwargs)
        elif agg_strat=='medoids':
            return self.medoids_pool(seeds,n,**kwargs)
        else:
            raise ValueError('{} not implemented'.format(agg_strat))


    def mean_pool(self,seeds, n=1000,**kwargs):
        vecs, found = self.get_vecs(seeds)

        if self.verbose:
            print('Averaging representation, returning pool of size {}'.format(n))
        if sum(found)==1:
            mean = vecs.reshape(-1,1)
        else:
            mean = np.mean(vecs,axis=0)
        # find most similar
        results = self.model.most_similar([mean],topn=int(n))
        pool_dist = [i[1] for i in results]
        pool_turi = [i[0].replace('spotify:track:','') for i in results]
        return pool_turi, pool_dist

    def centroids_pool(self,seeds, n=1000,n_clusters=10,**kwargs):
        vecs, found = self.get_vecs(seeds)
        # cluster size
        n_clusters = len(vecs) if n_clusters > len(vecs) else n_clusters
        if self.verbose:
            print('{}-means centroids, returning pool of size {}'.format(n_clusters,n))

        centroids = KMeans(n_clusters=n_clusters).fit(vecs).cluster_centers_
        # find most similar
        pool_dist=[]
        pool_turi=[]
        n_pick = ceil(n/n_clusters)
        for centroid in centroids:
            results = self.model.most_similar([centroid],topn=n_pick)
            pool_dist = pool_dist+[i[1] for i in results]
            pool_turi = pool_turi+[i[0].replace('spotify:track:','') for i in results]
        return pool_turi, pool_dist

    def medoids_pool(self,seeds, n=1000,n_clusters=10,**kwargs):
        vecs, found = self.get_vecs(seeds)
        # cluster size
        n_clusters = len(vecs) if n_clusters > len(vecs) else n_clusters
        if self.verbose:
            print('{}-Kmedoids , returning pool of size {}'.format(n_clusters,n))

        medoids = KMedoids(n_clusters=n_clusters).fit(vecs).cluster_centers_
        # find most similar
        pool_dist=[]
        pool_turi=[]
        n_pick = ceil(n/n_clusters)
        for medoid in medoids:
            results = self.model.most_similar([medoid],topn=n_pick)
            pool_dist = pool_dist+[i[1] for i in results]
            pool_turi = pool_turi+[i[0].replace('spotify:track:','') for i in results]
        return pool_turi, pool_dist
