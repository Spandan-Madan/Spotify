from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import os
from math import ceil
from tqdm import tqdm_notebook as tqdm


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
