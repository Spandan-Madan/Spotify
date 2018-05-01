from .base_feature import Feature
import numpy as np
import pandas as pd
from os.path import join as join_path
from sklearn.preprocessing import StandardScaler, RobustScaler
import sys
sys.path.append("..")
from data import compressed_pickle as cpick
DATA_PATH = '../data/interim'


class Popularity(Feature):

    def __init__(self, subset=''):
        Feature.__init__(self)
        afile = join_path(DATA_PATH, '{}popularity.pkl.bz2'.format(subset))
        self.pop = cpick.load(afile)
        self.preprocess()

    def preprocess(self):
        values = np.array([v for v in list(self.pop.values()) if not np.isnan(v)])
        self.preproc = StandardScaler().fit(values.reshape(-1,1))
        # impute
        mean_value = np.mean(values)
        for key,val in self.pop.items():
            if np.isnan(val):
                self.pop[key]=mean_value

        return

    def transform(self, turis=None):
        if turis is None:
            vals = np.array([v for v in list(self.pop.values())])
        else:
            vals = np.array([self.pop[turi] for turi in turis])

        return self.preproc.transform(vals.reshape(-1, 1))

