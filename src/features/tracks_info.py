import numpy as np
import pandas as pd
from os.path import join as join_path
import sys
sys.path.append("..")
from data.data_utils import read_playlist
from data import compressed_pickle as cpick
DATA_PATH = '../data/interim'
PLIST_PATH = '../data/raw'


class TrackInfo(object):

    def __init__(self, subset=''):
        #self.data
        #../data/interim/5k_playlist2track_ints.pkl.bz2
        #self.datafile = join_path(DATA_PATH, '5k_playlist2track_ints.pkl.bz2')
        afile = join_path(DATA_PATH, '{}playlist2track_ints.pkl.bz2'.format(subset))
        self.plists = cpick.load(afile)
        afile = join_path(DATA_PATH, '{}track_int2track_uri.pkl.bz2'.format(subset))
        self.tint2turi = cpick.load(afile)
        afile = join_path(DATA_PATH, 'track_uri2track_name.pkl.bz2')
        self.turi2tname = cpick.load(afile)
        afile = join_path(DATA_PATH, 'track_uri2artist_uri.pkl.bz2')
        self.turi2auri = cpick.load(afile)
        afile = join_path(DATA_PATH, 'artist_uri2artist_name.pkl.bz2')
        self.auri2aname = cpick.load(afile)
    # def track_df(self, ids):
    #     return self.df.ilocs[ids]

    def get_playlist_turi(self, pid):
        plist = [self.tint2turi[i] for i in self.plists[pid]]
        return plist

    def get_playlist_tints(self, pid):
        return self.plists[pid]

    def get_playlist_auri(self, pid):
        plist = [self.turi2auri[self.tint2turi[i]] for i in self.plists[pid]]
        return plist

    def get_playlist(self, pid):
        return self.get_playlist_turi(pid),self.get_playlist_tints(pid),self.get_playlist_auri(pid)


    def uri2track_info(self,uri):
        tname = self.turi2tname[uri]
        aname = self.auri2aname[self.turi2auri[uri]]
        return '{} - {}'.format(tname,aname)



