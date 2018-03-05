from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import os
from math import ceil
from tqdm import tqdm_notebook as tqdm
import fuzzywuzzy as fuzz
import numpy as np
from data_utils import normalize_name


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def setup_spotify():
    os.environ['SPOTIPY_CLIENT_ID'] = '55a76a654fe14f739a30dcc5fbfa055e'
    os.environ['SPOTIPY_CLIENT_SECRET'] = '7108a6157a8a437482f073496dd9c2a7'
    os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost/'
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp.trace = False
    return sp


def df_chunk_apply(df, chunk_size, func):
    sp = setup_spotify()
    n_chunks = ceil(len(df) / chunk_size)
    rows = []
    for chunk in tqdm(chunker(df, chunk_size), total=n_chunks):
        rows += func(chunk, sp)
    return rows


def replace_none(rows):
    # check for None
    dummy_feature = {k: None for k, v in rows[0].items()}
    for indx, row in enumerate(rows):
        if not isinstance(row, dict):
            rows[indx] = dummy_feature.copy()
    return


def unique_genres(df, genre_col='genres'):
    genres = set()
    for g in df[genre_col]:
        genres = genres.union(g)
    return list(genres)


def fix_track_uri(df, subset, sp):
    n_fixed = 0
    for indx, row in df[subset].iterrows():
        tracks = sp.search(row['title'], limit=10,
                           offset=0, type='track', market=None)
        print(row['title'], row['artist'])
        best_uri = None
        best_ll = -np.inf
        best_name = None
        for t in tracks['tracks']['items']:
            q_name = normalize_name(t['artists'][0]['name'])
            ll = fuzz.ratio(row['artist'], q_name)
            if ll > best_ll:
                best_uri = t['uri']
                best_ll = ll
                best_name = q_name

        print(best_ll, best_name, best_uri)
        if best_ll > 60:
            df.at[indx, 'uri'] = best_uri
            n_fixed += 1
    print('Fixed {} out of {}'.format(n_fixed, sum(subset)))
    return df


def fix_artist_uri(df, subset, sp):
    n_fixed = 0
    for indx, row in df[subset].iterrows():
        arts = sp.search(row['artist'], limit=10, offset=0,
                         type='artist', market=None)
        print(row['title'], row['artist'])
        best_uri = None
        best_ll = -np.inf
        best_name = None
        for a in arts['artists']['items']:
            q_name = normalize_name(a['name'])
            ll = fuzz.ratio(row['artist'], q_name)
            if ll > best_ll:
                best_uri = a['uri']
                best_ll = ll
                best_name = q_name

        print(best_ll, best_name, best_uri)
        if best_ll > 60:
            df.at[indx, 'artist_uri'] = best_uri
            n_fixed += 1
    print('Fixed {} out of {}'.format(n_fixed, sum(subset)))
    return df
