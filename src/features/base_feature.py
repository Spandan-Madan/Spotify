from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


def sortlists_by_dist(pool, dist):
    a, b = (list(t) for t in zip(*sorted(zip(dist, pool), reverse=True)))
    return b, a


def labels2proportions(labels, n):
    labels_n = len(labels)
    w_piece = n / float(len(labels))
    weights = [(key, int(item * w_piece))
               for key, item in Counter(labels).items()]
    return [int(w) for i, w in sorted(weights, key=lambda tup: tup[0])]


def ncluster_ruleofthumb(n):
    return int(ceil(np.sqrt(n / 2)))


class Feature(object):

    def __init__(self, subset=''):
        '''
        Feature constructor, will load all relevant data, if using the 5k subset, use the string "5k-"
        Input: can be whatever you need (dimention of embedding, folder,etc)
        '''

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
        D = pairwise_distances(mean_vec, pool_vec).reshape(-1, 1)
        return D.ravel()

    def wcentroids_distance(self, seed_vec, pool_vec, n_clusters=None, **kwargs):
        # cluster size
        n_clusters = ncluster_ruleofthumb(
            len(vecs)) if n_clusters is None else n_clusters
        n_clusters = len(vecs) if n_clusters > len(vecs) else n_clusters

        centers, labels, inertia = k_means(vecs, n_clusters)
        props = labels2proportions(labels, n)

        # find most similar
        pool_dist = []
        pool_turi = []
        for center, n_pick in zip(centers, props):
            D = pairwise_distances(mean_vec, pool_vec).reshape(-1, 1)

            results = self.model.wv.similar_by_vector(center, topn=n_pick)
            pool_dist = pool_dist + [i[1] for i in results]
            pool_turi = pool_turi + \
                [i[0].replace('spotify:track:', '') for i in results]
        pool_turi, pool_dist = sortlists_by_dist(pool_turi, pool_dist)
        return pool_turi, pool_dist

    def split_distance(self, seed_vec, pool_vec, **kwargs):

        # find most similar
        pool_dist = []
        pool_turi = []
        n_pick = ceil(n / len(seed_vec))
        for vec in vecs:
            results = self.model.wv.similar_by_vector(vec, topn=n_pick)
            pool_dist = pool_dist + [i[1] for i in results]
            pool_turi = pool_turi + \
                [i[0].replace('spotify:track:', '') for i in results]
        pool_turi, pool_dist = sortlists_by_dist(pool_turi, pool_dist)
        return pool_turi, pool_dist

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
            return self.mean_distance(seed_vec, pool_vec)

        return
