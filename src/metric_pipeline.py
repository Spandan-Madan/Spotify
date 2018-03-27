from data.data_utils import read_playlist
from data.metrics import r_precision, cosine_sim_closest, cosine_sim_top
import pandas as pd
import numpy as np
from tqdm import tqdm
from features.audio_features import AudioFeatures
from features.tracks_info import TrackInfo

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
x_pred = []
print('Test on playlist {:d}'.format(pid))
for i in tqdm(range(n_songs), total=n_songs):
    x_partial = xp[:i, :]
    x_query = xp[i - 1, :]
    ind, sim = cosine_sim_top(X, x_query, 0.95)
    if len(ind) == 0:
        ind, sim = cosine_sim_closest(X, x_query, 1)
    # x_pred[i*n_top:(i+1)*n_top]=ind
    x_pred.append(ind)

    x_next = xp[i, :]
x_pred = np.array(x_pred)

score = r_precision(p_ids, list(np.concatenate(x_pred)))
print(score)
