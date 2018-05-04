from os.path import join as path_join
import numpy as np
from gensim.models import Word2Vec
DATA_PATH = '../data/pooling'
from sklearn.cluster import KMeans
from math import ceil
from sklearn.cluster import k_means
from medoids import k_medoids
from collections import Counter

def sortlists_by_dist(pool,dist):
    a,b=(list(t) for t in zip(*sorted(zip(dist,pool), reverse=True)))
    return b,a

def labels2proportions(labels,n):
    labels_n = len(labels)
    w_piece = n/float(len(labels))
    weights = [(key,int(item*w_piece))  for key,item in Counter(labels).items()]
    return [int(w) for i,w in sorted(weights, key=lambda tup: tup[0])]

def ncluster_ruleofthumb(n):
    return int(ceil(np.sqrt(n/2)))

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
        elif agg_strat=='wcentroids':
            return self.wcentroids_pool(seeds,n,**kwargs)
        elif agg_strat=='wmedoids':
            return self.wmedoids_pool(seeds,n,**kwargs)
        elif agg_strat=='split':
            return self.split_pool(seeds,n,**kwargs)
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
        results = self.model.wv.similar_by_vector(mean,topn=int(n))
        pool_dist = [i[1] for i in results]
        pool_turi = [i[0].replace('spotify:track:','') for i in results]
        return pool_turi, pool_dist

    def centroids_pool(self,seeds, n=1000,n_clusters=None,**kwargs):
        vecs, found = self.get_vecs(seeds)
        # cluster size
        n_clusters = ncluster_ruleofthumb(len(vecs)) if n_clusters is None else n_clusters
        n_clusters = len(vecs) if n_clusters > len(vecs) else n_clusters

        if self.verbose:
            print('{}-means centroids, returning pool of size {}'.format(n_clusters,n))

        centers,labels,inertia = k_means(vecs,n_clusters)
        # find most similar
        pool_dist=[]
        pool_turi=[]
        n_pick = ceil(n/n_clusters)
        for center in centers:
            results = self.model.wv.similar_by_vector(center,topn=n_pick)
            pool_dist = pool_dist+[i[1] for i in results]
            pool_turi = pool_turi+[i[0].replace('spotify:track:','') for i in results]
        pool_turi, pool_dist = sortlists_by_dist(pool_turi,pool_dist)
        return pool_turi, pool_dist

    def wcentroids_pool(self,seeds, n=1000,n_clusters=None,**kwargs):
        vecs, found = self.get_vecs(seeds)
        # cluster size
        n_clusters = ncluster_ruleofthumb(len(vecs)) if n_clusters is None else n_clusters
        n_clusters = len(vecs) if n_clusters > len(vecs) else n_clusters
        if self.verbose:
            print('{}-means weighted, returning pool of size {}'.format(n_clusters,n))

        centers,labels,inertia = k_means(vecs,n_clusters)
        props = labels2proportions(labels,n)

        # find most similar
        pool_dist=[]
        pool_turi=[]
        for center, n_pick in zip(centers,props):
            results = self.model.wv.similar_by_vector(center,topn=n_pick)
            pool_dist = pool_dist+[i[1] for i in results]
            pool_turi = pool_turi+[i[0].replace('spotify:track:','') for i in results]
        pool_turi, pool_dist = sortlists_by_dist(pool_turi,pool_dist)
        return pool_turi, pool_dist

    def medoids_pool(self,seeds, n=1000,n_clusters=None,**kwargs):
        vecs, found = self.get_vecs(seeds)
        # cluster size
        n_clusters = ncluster_ruleofthumb(len(vecs)) if n_clusters is None else n_clusters
        n_clusters = len(vecs) if n_clusters > len(vecs) else n_clusters
        if self.verbose:
            print('{}-medoids , returning pool of size {}'.format(n_clusters,n))

        medoids, labels = k_medoids(vecs,n_clusters=n_clusters)
        # find most similar
        pool_dist=[]
        pool_turi=[]
        n_pick = ceil(n/n_clusters)
        for medoid in medoids:
            results = self.model.wv.similar_by_vector(medoid,topn=n_pick)
            pool_dist = pool_dist+[i[1] for i in results]
            pool_turi = pool_turi+[i[0].replace('spotify:track:','') for i in results]
        pool_turi, pool_dist = sortlists_by_dist(pool_turi,pool_dist)
        return pool_turi, pool_dist

    def wmedoids_pool(self,seeds, n=1000,n_clusters=None,**kwargs):
        vecs, found = self.get_vecs(seeds)
        # cluster size
        n_clusters = ncluster_ruleofthumb(len(vecs)) if n_clusters is None else n_clusters
        n_clusters = len(vecs) if n_clusters > len(vecs) else n_clusters
        if self.verbose:
            print('{}-medoids weighted, returning pool of size {}'.format(n_clusters,n))

        medoids, labels = k_medoids(vecs,n_clusters=n_clusters)
        props = labels2proportions(labels,n)
        # find most similar
        pool_dist=[]
        pool_turi=[]
        for medoid, n_pick in zip(medoids,props):
            results = self.model.wv.similar_by_vector(medoid,topn=n_pick)
            pool_dist = pool_dist+[i[1] for i in results]
            pool_turi = pool_turi+[i[0].replace('spotify:track:','') for i in results]
        pool_turi, pool_dist = sortlists_by_dist(pool_turi,pool_dist)
        return pool_turi, pool_dist


    def split_pool(self,seeds, n=1000,n_clusters=10,**kwargs):
        vecs, found = self.get_vecs(seeds)

        # find most similar
        pool_dist=[]
        pool_turi=[]
        n_pick = ceil(n/n_clusters)
        for vec in vecs:
            results = self.model.wv.similar_by_vector(vec,topn=n_pick)
            pool_dist = pool_dist+[i[1] for i in results]
            pool_turi = pool_turi+[i[0].replace('spotify:track:','') for i in results]
        pool_turi, pool_dist = sortlists_by_dist(pool_turi,pool_dist)
        return pool_turi, pool_dist
