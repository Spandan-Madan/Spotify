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
        "mpd.slice.") and x.endswith(".json")
    good_files = [f for f in filenames if is_playlist(f)]
    if rand:
        shuffle(good_files)
    else:
        good_files =sorted(good_files)

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


if __name__ == '__main__':
    print('load me as a module!')
