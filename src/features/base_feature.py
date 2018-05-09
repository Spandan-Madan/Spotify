from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from sklearn.cluster import k_means
from medoids import k_medoids
from math import ceil

def ncluster_ruleofthumb(n):
    return int(ceil(np.sqrt(n / 2)))

class Feature(object):

    def __init__(self, subset=''):
        '''
        Feature constructor, will load all relevant data, if using the 5k subset, use the string "5k-"
        Input: can be whatever you need (dimention of embedding, folder,etc)
        '''
        self.metric = 'cosine'
        return

    def transform(self, turis):
        '''
        Convert track a list of turis to a vector representation.
        Input: turis is a list of strings of length n
        Output: will return a vector of shape (n_dim,n)
        '''

        return

    def mean_distance(self, seed_vec, pool_vec, **kwargs):
        mean_vec = np.mean(seed_vec, axis=0).reshape(1, -1)
        D = pairwise_distances(
            mean_vec, pool_vec, metric=self.metric).reshape(-1, 1)
        return D.ravel()

    def all_distance(self, seed_vec, pool_vec, **kwargs):
        D = pairwise_distances(seed_vec, pool_vec, metric=self.metric)
        return np.min(D, axis=0)

    def centroid_distance(self, seed_vec, pool_vec, n_clusters = None, **kwargs):
        n_clusters = ncluster_ruleofthumb(
            len(seed_vec)) if n_clusters is None else n_clusters
        n_clusters = len(seed_vec) if n_clusters > len(
            seed_vec) else n_clusters

        centroid_vec, labels, inertia = k_means(seed_vec, n_clusters)

        D = pairwise_distances(centroid_vec, pool_vec, metric=self.metric)
        return np.min(D, axis=0)

    def medoid_distance(self, seed_vec, pool_vec, n_clusters = None, **kwargs):
        n_clusters = ncluster_ruleofthumb(
            len(seed_vec)) if n_clusters is None else n_clusters
        n_clusters = len(seed_vec) if n_clusters > len(
            seed_vec) else n_clusters

        medoid_vec, labels = k_medoids(seed_vec, n_clusters=n_clusters)

        D = pairwise_distances(medoid_vec, pool_vec, metric=self.metric)
        return np.min(D, axis=0)

    def distance_between_sets(self, seeds, pool, strat, **kwargs):
        '''
        Calculate the distance between two sets of tracks (seeds) and (pools). Will return a numpy array of distances.
        Input: seeds is a list of track_uris (strings)
               pool is a list
        Output: will return a vector of shape (n_dim,n)
        '''
        seed_vec = self.transform(seeds)
        pool_vec = self.transform(pool)
        if strat == 'mean':
            return self.mean_distance(seed_vec, pool_vec, **kwargs)
        elif strat == 'all':
            return self.all_distance(seed_vec, pool_vec, **kwargs)
        elif strat == 'centroid':
            return self.centroid_distance(seed_vec, pool_vec, **kwargs)
        elif strat == 'medoid':
            return self.medoid_distance(seed_vec, pool_vec, **kwargs)
        else:
            raise ValueError('{} nor implemented'.format(strat))
        return
