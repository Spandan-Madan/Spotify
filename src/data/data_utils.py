# -*- coding: utf-8 -*-
import json
try:
    ipy_str = str(type(get_ipython()))
    from tqdm import tqdm_notebook as tqdm
except:
    from tqdm import tqdm
import os
from random import shuffle
import re
import datetime
import functools, itertools, operator

def product_size(iters):
    return functools.reduce(operator.mul, map(len, iters), 1)

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def to_date(epoch):
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")


def process_mpd(path, func, results, max_n=None, rand=False):
    filenames = os.listdir(path)

    def is_playlist(x): return x.startswith(
        "mpd") and x.endswith(".json")
    good_files = [f for f in filenames if is_playlist(f)]
    if rand:
        shuffle(good_files)
    else:
        good_files = sorted(good_files)

    if max_n is not None:
        good_files = good_files[:min(max_n, len(good_files))]

    for filename in tqdm(good_files):
        fullpath = os.sep.join((path, filename))
        with open(fullpath) as f:
            js = f.read()
        mpd_slice = json.loads(js)
        for playlist in mpd_slice['playlists']:
            func(playlist, results)

    return

def process_mpd2(path, func_pl, results_pl, func_json, results_json, max_n=None, rand=False):
    filenames = os.listdir(path)

    def is_playlist(x): return x.startswith(
        "mpd") and x.endswith(".json")
    good_files = [f for f in filenames if is_playlist(f)]
    if rand:
        shuffle(good_files)
    else:
        good_files = sorted(good_files)

    if max_n is not None:
        good_files = good_files[:min(max_n, len(good_files))]

    for n_file,filename in tqdm(enumerate(good_files),total=len(good_files)):
        fullpath = os.sep.join((path, filename))
        with open(fullpath) as f:
            js = f.read()
        mpd_slice = json.loads(js)
        for playlist in mpd_slice['playlists']:
            func_pl(playlist, results_pl)
        func_json(n_file,results_json,results_pl)

    return



def read_playlist(path, pid):
    low_id = pid - pid % 1000
    hi_id = low_id + 999
    name = 'mpd.slice.{:d}-{:d}.json'.format(low_id, hi_id)
    filename = os.path.join(path, name)
    with open(filename) as f:
        js = f.read()
    mpd_slice = json.loads(js)
    for playlist in mpd_slice['playlists']:
        if playlist['pid'] == pid:
            return playlist
    raise ValueError('Did not find playlist {} in {}'.format(pid, filename))
    return


if __name__ == '__main__':
    print('load me as a module!')
