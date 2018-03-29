# sportify specific
from data.metrics_track import r_precision, cosine_sim_closest, cosine_sim_top, NDCG
from visualization.plot_utils import write_latex_table
# general
import pandas as pd
import numpy as np
from tqdm import tqdm
from features.audio_features import AudioFeatures
from features.tracks_info import TrackInfo
from collections import OrderedDict

k_range = [1, 5, 10, 25, 100]
max_picks = 500

tracks = TrackInfo()
# audio features
af = AudioFeatures()
X = af.transform()
n_cols = X.shape[1]

# load playlist
pid = 194
p_ids = tracks.get_playlist_ids(pid)
xp = af.subset(p_ids)
n_songs = len(p_ids)

suggest_df = pd.DataFrame()
n_top = 10
x_pred = OrderedDict()
stats = []
print('Test on playlist {:d}'.format(pid))
for k in k_range:
    x_pred[k] = []
    k_picks = int(max_picks / k)
    for i in range(k):
        x_query = xp[i - 1, :]
        ind, sim = cosine_sim_closest(X, x_query, k_picks)
        x_pred[k] = x_pred[k] + list(ind)
    subset_ids = p_ids[k:]
    result = OrderedDict()
    result['k'] = k
    result['r precision'] = r_precision(subset_ids, x_pred[k])
    result['NDGC'] = NDCG(subset_ids, x_pred[k])
    stats.append(result)

sdf = pd.DataFrame(stats)
print(sdf)

write_latex_table(sdf, 'metrics', adir='.', render=True)
