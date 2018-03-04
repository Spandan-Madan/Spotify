import numpy as np
from scipy.spatial.distance import cosine, hamming
from scipy.stats import pearsonr


def inv_cosine(xi, xj):
    dij = cosine(xi, xj)
    val = 1.0 / dij
    return val


def inv_hamming(xi, xj):
    dij = hamming(xi, xj)
    val = 1.0 / dij
    return val


def pearson(xi, xj):
    return pearsonr(xi, xj)[0]


def diversity(plist, dist_f, norm_f):
    """ diversity of a playlist

    Common choices for dist_f are inverse cosine similarity, inverse
    Pearson correlation, or Hamming distance
    """
    n = len(plist)
    sum_d = np.sum([dist_f(plist[i], plist[j])
                    for i in range(n) for j in range(i + 1, n)])

    p_norm = norm_f(plist)
    val = sum_d / (p_norm * (p_norm - 1))

    return val

