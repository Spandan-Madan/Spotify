import numpy as np
import pandas as pd
from os.path import join as join_path
import sys
sys.path.append("..")
from data.data_utils import read_playlist
from data import compressed_pickle as cpick
DATA_PATH = '../data/interim'
PLIST_PATH = '../data/raw'

def data_path(a):
    return join_path(DATA_PATH,a)

class TrackInfo(object):

    def __init__(self, subset=''):

        afile = data_path('{}playlist2track_ints.pkl.bz2'.format(subset))
        self.plists = cpick.load(afile)
        # track information
        afile = data_path('{}track_int2track_uri.pkl.bz2'.format(subset))
        self.tint2turi = cpick.load(afile)
        afile = data_path('track_uri2track_name.pkl.bz2')
        self.turi2tname = cpick.load(afile)
        afile = data_path('track_uri2artist_uri.pkl.bz2')
        self.turi2auri = cpick.load(afile)
        # artist information
        afile = data_path('artist_uri2artist_name.pkl.bz2')
        self.auri2aname = cpick.load(afile)
        afile = data_path('artist_name2artist_uri.pkl.bz2')
        self.aname2auri = cpick.load(afile)
        afile = data_path('artist_uri2track_uri.pkl.bz2')
        self.auri2turi = cpick.load(afile)

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
        return self.get_playlist_turi(pid),self.get_playlist_auri(pid)

    def track_info2uris(self,track_name,artist_name):
        auris = self.aname2auri[artist_name]
        for auri in auris:
            tracks = [(t,self.turi2tname[t])  for t in self.auri2turi[auri]]
            for turi,tname in tracks:
                if tname == track_name:
                    return turi,auri

        raise ValueError('Cannot find {} - {}'.format(track_name,artist_name))
        return

    def uri2track_info(self,uri):
        tname = self.turi2tname[uri]
        aname = self.auri2aname[self.turi2auri[uri]]
        return '{} - {}'.format(tname,aname)



