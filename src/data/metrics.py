import numpy as np
from scipy.spatial.distance import cosine, hamming
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim_closest(X, x_query, n=1):
    lim = n + 1
    sim = cosine_similarity(X, x_query.reshape(1, -1)).ravel()
    ind = np.argpartition(sim, -lim)[-lim:]
    ind = ind[np.argsort(sim[ind])]
    return ind[:-1], sim[ind[:-1]]


def cosine_sim_top(X, x_query, tol=0.95):
    sim = cosine_similarity(X, x_query.reshape(1, -1)).ravel()
    lim = sum(sim > tol)
    ind = np.argpartition(sim, -lim)[-lim:]
    ind = ind[np.argsort(sim[ind])]
    return ind[:-1], sim[ind[:-1]]


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


def inv_pearson(xi, xj):
    return 1.0 / pearson(xi, xj)


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


def dcg_from_ranking(y_true, ranking):
   """Discounted cumulative gain (DCG) at rank k
   Parameters
   ----------
   y_true : array-like, shape = [n_samples]
       Ground truth (true relevance labels).
   ranking : array-like, shape = [k]
       Document indices, i.e.,
           ranking[0] is the index of top-ranked document,
           ranking[1] is the index of second-ranked document,
           ...
   k : int
       Rank.
   Returns
   -------
   DCG @k : float
   """
   y_true = np.asarray(y_true)
   ranking = np.asarray(ranking)
   rel = y_true[ranking]
   gains = 2 ** rel - 1
   discounts = np.log2(np.arange(len(ranking)) + 2)
   return np.sum(gains / discounts)

def NDCG_1(y_true, ranking):
   """Normalized discounted cumulative gain (NDCG) at rank k
   Parameters
   ----------
   y_true : array-like, shape = [n_samples]
       Ground truth (true relevance labels).
   ranking : array-like, shape = [k]
       Document indices, i.e.,
           ranking[0] is the index of top-ranked document,
           ranking[1] is the index of second-ranked document,
           ...
   k : int
       Rank.
   Returns
   -------
   NDCG @k : float
   """
   y_true = np.array(y_true)
   ranking = np.array(ranking)
   k = len(ranking)
   best_ranking = np.argsort(y_true)[::-1]
   best = dcg_from_ranking(y_true, best_ranking)
   return dcg_from_ranking(y_true, ranking) / best


def IDCG(true, pred):
    n_common = len(set(true).intersection(set(pred)))
    return 1 + np.sum([1.0 / np.log2(i) for i in range(2, n_common + 1)])

def DCG(true, pred):

    relevant = set(true).intersection(set(pred))
    n_common = len(set(true).intersection(set(pred)))
    if n_common == 0:
        return 0.0
    else:
        ranks = [i for i,p in enumerate(pred) if p in relevant]
        score = 0.0
        for order, rank in enumerate(ranks):
            score += float(rank)/float(len(pred)) / np.log2((order + 2))
        return score


def NDCG(true, pred):
    return DCG(true, pred) / IDCG(true, pred)


def recall(true, pred):
    """
    pred: a single prediction playlist
    true: ground truth

    """
    return len(set(true).intersection(set(pred))) / float(len(true))


def recommended_song_click(true, pred):
    """
    pred: list of predictions
    true: ground truth
    """
    cnt = 1
    for predictions in pred:
        if true in predictions:
            break
        else:
            cnt += 1

    return cnt / 10 * 1.0

def r_precision(true, pred):
    """
    pred: a single, ranked prediction playlist
    true: ground truth
    """
    gt_length = len(true)
    top_preds = pred[:gt_length]
    return len(set(true).intersection(set(top_preds))) / float(len(true))
