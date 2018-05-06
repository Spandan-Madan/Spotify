from .base_feature import Feature
import numpy as np
import pandas as pd
from os.path import join as join_path
from sklearn.preprocessing import StandardScaler, RobustScaler
import sys
sys.path.append("..")
from data import compressed_pickle as cpick
DATA_PATH = '../data/interim'


class AudioFeatures(Feature):

    def __init__(self, subset=''):
        Feature.__init__(self)
        afile = join_path(DATA_PATH, '{}audio_features.pkl.bz2'.format(subset))
        self.df = cpick.load(afile)
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
                    'valence': StandardScaler()
                    }
        #,            'popularity': StandardScaler()
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
        return

    def transform(self, turis=None):
        if turis is None:
            df = self.df
        else:
            df = self.df.loc[turis]
        # get X vector
        X = np.zeros((len(df), len(self.cols)))
        for i, pair in enumerate(self.preprocs.items()):
            key, pre = pair
            X[:, i] = pre.transform(df[key].values.reshape(-1, 1)).ravel()
        print (X.shape)
        return X

