import numpy as np
import pandas as pd
from os.path import join as join_path
import sys
sys.path.append("..")
from data.data_utils import read_playlist

DATA_PATH = '../data/interim'
PLIST_PATh = '../data/raw'


class TrackInfo(object):

    def __init__(self, subset=''):
        self.datafile = join_path(DATA_PATH, '5k_track_uri.csv')
        self.df = pd.read_csv(self.datafile)

    def track_df(self, ids):
        return self.df.ilocs[ids]

    def get_playlist_ids(self, pid):
        pfile = join_path(PLIST_PATh, '5k_subset')
        plist = read_playlist(pfile, pid)
        n_songs = len(plist['tracks'])
        ids = np.zeros(n_songs, int)
        for track in plist['tracks']:
            row = self.df[self.df['uri'] == track['track_uri']].iloc[0]
            ids[track['pos']] = int(row['csv_id'])
        return ids
