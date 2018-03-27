import numpy as np
import pandas as pd
from os.path import join as join_path
from sklearn.preprocessing import StandardScaler, RobustScaler

DATA_PATH = '../data/interim'


class AudioFeatures(object):

    def __init__(self, subset=''):
        self.datafile = join_path(DATA_PATH, '5k_track_audiofeatures.csv')
        self.df = pd.read_csv(self.datafile)
        self.preprocess(self.df)

    def preprocess(self, df):

        df['duration_ms'] = df['duration_ms'].apply(
            lambda x: np.clip(x / 1000.0, 0, 600))

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
        # impute
        cols = list(preprocs.keys())
        f_df = df[cols]
        #n_missing = sum([sum(f_df[c].isnull()) for c in cols])
        #print('{:d} missing values, imputed with mean'.format(n_missing))
        # mean inputation
        means = {key: np.mean(f_df[key].dropna().values) for key in cols}
        # update variables
        self.df = f_df.fillna(value=means)
        self.preprocs = preprocs
        self.cols = list(self.preprocs.keys())
        # fit transformation
        for key, pre in preprocs.items():
            pre.fit(self.df[key].values.reshape(-1, 1))

    def transform(self, df=None):
        if df is None:
            df = self.df
        # get X vector
        X = np.zeros((len(df), len(self.cols)))
        for i, pair in enumerate(self.preprocs.items()):
            key, pre = pair
            X[:, i] = pre.transform(df[key].values.reshape(-1, 1)).ravel()
        return X

    def subset(self, ids):
        return self.transform(self.df.loc[ids])
