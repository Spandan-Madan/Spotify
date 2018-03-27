from data.data_utils import read_playlist
from data.metrics import r_precision, cosine_sim_closest, cosine_sim_top
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

track_df = pd.read_csv('../data/interim/5k_track_uri.csv')
# audio features
df = pd.read_csv('../data/interim/5k_track_audiofeatures.csv')
df['duration_ms'] = df['duration_ms'].apply(
    lambda x: np.clip(x / 1000.0, 0, 600))
# stack
preprocs = {'acousticness': StandardScaler(),
            'danceability': StandardScaler(),
            'duration_ms': RobustScaler(),
            'energy': StandardScaler(),
            'liveness': StandardScaler(),
            'loudness': RobustScaler(),
            'speechiness': RobustScaler(),
            'tempo': StandardScaler(),
            'valence': StandardScaler(),
            'popularity': StandardScaler()
            }
cols = list(preprocs.keys())
f_df = df[cols]
n_missing = sum([sum(f_df[c].isnull()) for c in cols])
print('{:d} missing values, imputed with mean'.format(n_missing))
# mean inputation
means = {key: np.mean(f_df[key].dropna().values) for key in cols}
f_df = f_df.fillna(value=means)
# get X vector
X = np.zeros((len(f_df), len(cols)))
for i, pair in enumerate(preprocs.items()):
    key, pre = pair
    X[:, i] = pre.fit_transform(f_df[key].values.reshape(-1, 1)).ravel()
print('X has {}'.format(X.shape))
# load playlist
pid = 194
plist = read_playlist('../data/raw/5k_subset', pid)
n_songs = len(plist['tracks'])
print(n_songs)
xp = np.zeros((n_songs, len(cols)))
xp_ids = np.zeros(n_songs, int)
for track in plist['tracks']:
    row = track_df[track_df['uri'] == track['track_uri']].iloc[0]
    indx = row['csv_id']
    xp_ids[track['pos']] = int(indx)
    xp[track['pos'], :] = X[indx, :]


suggest_df = pd.DataFrame()
n_top = 10
x_pred = np.zeros(n_songs * n_top, int)
x_pred = []
for i in tqdm(range(n_songs), total=n_songs):

    x_partial = xp[:i, :]
    x_query = xp[i - 1, :]
    ind, sim = cosine_sim_top(X, x_query, 0.95)
    if len(ind) == 0:
        ind, sim = cosine_sim_closest(X, x_query, 1)

    # x_pred[i*n_top:(i+1)*n_top]=ind
    x_pred.append(ind)

    top_suggested_track = track_df.iloc[ind[0]]
    suggest_df = suggest_df.append(top_suggested_track)
    x_next = xp[i, :]

score = r_precision(xp_ids, list(np.concatenate(x_pred)))
print(score)
#html_header('Score = {}'.format(score))
#html_header('Suggested')
#display(suggest_df[['artist', 'title']])
#html_header('Real')
#display(t_df.iloc[xp_ids][['artist', 'title']])
