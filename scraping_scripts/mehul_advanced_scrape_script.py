import pandas as pd
import numpy as np
import json
import urllib.request as urllib2
import requests
from dicttoxml import dicttoxml
from lxml import etree
from IPython.display import display
import re
from bs4 import BeautifulSoup
import csv

temp = pd.read_csv('all_unique_songs.csv')
print(temp.shape)


import unicodedata

def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())

def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)


client_access_token = 'nwNMYfaDcBVXGluE8SsY_h1CX3Jy0BSEDRHgZ6bGUzjNc1_lZ97mluDwDTRov2Wj'

all_songs = list(zip(temp['artist_name'],temp['track_name']))


_URL_API = "https://api.genius.com/"
_URL_SEARCH = "search?q="

mehul = 187442

fail_count = 0
# results = []
idx = mehul #0
df = pd.DataFrame()
lim = 1000 #00

print('Starting ', idx)
start_idx = mehul
file = open(str(idx)+'.csv', 'w')
writer = csv.writer(file)
writer.writerow(['song','artist','query_id','genius_id','about','artist_speak','trivia','annotation'])

for artist, song in all_songs[mehul:200000]:
    # Separating artist information
    full_song = song
    only_song = [song.split('-')[0], song.split('(')[0]]
    search_terms = [artist+' '+only_song[0], artist+' '+only_song[1], artist+' '+song, only_song[0], only_song[1], artist] # Seven cases of search to improve search
    song_res = [song, artist]  # song, artist, Query that worked, Genius ID,
    song_id = None

    # Finding the song
    try:
        qid_that_worked = -1
        for term in search_terms:
            if song_id is not None:
                break
            querystring = _URL_API + _URL_SEARCH + urllib2.quote(term)
            request = urllib2.Request(querystring)
            request.add_header("Authorization", "Bearer " + client_access_token)
            request.add_header("User-Agent", "")

            response = urllib2.urlopen(request, timeout=2)
            raw = response.read()
            # print(term, raw)
            # Getting song id of first song from response
            json_obj = json.loads(raw)
            for i in range(len(json_obj['response']['hits'])):
                primary_artist = normalize_caseless(json_obj['response']['hits'][i]['result']['primary_artist']['name'])
                pa = re.sub(r'[^\w\s]','',primary_artist)
                full_title = normalize_caseless(json_obj['response']['hits'][i]['result']['full_title'])
                if (primary_artist == artist or pa == artist) and (only_song[0] in full_title or only_song[1] in full_title):
                    song_id = json_obj['response']['hits'][i]['result']['id']
                    break
            qid_that_worked+=1

        # If song not found for any query, count as excption
        if song_id is None:
            raise ValueError()

        # Add details of matched song
        print(qid_that_worked, song_id, full_title)
        song_res.append(qid_that_worked)
        song_res.append(song_id)
    except:
        print('Song retrieval failed')
        print(artist,song,'\n')
        song_res.append(-1)
        song_res.append(None)


    if song_id is not None:
        try:
            # Querying for song ID and HTML page
            querystring = "https://api.genius.com/songs/" + str(song_id)
            request = urllib2.Request(querystring)
            request.add_header("Authorization", "Bearer " + client_access_token)
            request.add_header("User-Agent", "")
            response = urllib2.urlopen(request, timeout=3)
            raw = response.read()
            json_obj = json.loads(raw)['response']['song']
            about = json_obj['description']['dom']
            html_path = json_obj['path'] # print(html_path)

            # Getting data from song HTML page
            page_url = "http://genius.com" + html_path
            page = requests.get(page_url)
            html = BeautifulSoup(page.text, "html.parser")

            # Getting song description from page
            labels = html.find_all(class_='annotation_label')
            detail = html.find_all(name = 'div', class_='rich_text_formatting')
            about, artist_speak, trivia = None, None, None
            if len(labels)>0: about = detail[0].get_text()
            if len(labels)>1: artist_speak = detail[1].get_text()
            if len(labels)>2:
                trivia = ''
                for l,d in zip(labels[2:], detail[2:]):
                    trivia += l.get_text() + d.get_text() + '\n'

            song_res.append(about)
            song_res.append(artist_speak)
            song_res.append(trivia)


            # Getting annotations
            ann_text = ''
            all_ann_tags = html.find_all(name = 'div', class_='lyrics')[0].find_all(name = 'a') # annotation-fragment')
            for ann_tag in all_ann_tags:
                ann = ann_tag['annotation-fragment']

                # Getting text from annotation
                aa = "https://api.genius.com/annotations/"+str(ann)
                request = urllib2.Request(aa)
                request.add_header("Authorization", "Bearer " + client_access_token)
                request.add_header("User-Agent", "")
                response = urllib2.urlopen(request, timeout=3)
                raw = json.loads(response.read())

                desc = ''
                xml = dicttoxml(raw['response']['annotation']['body']['dom']['children'], custom_root='children')
                q = etree.fromstring(xml)
                for child in q.findall(".//item[@type='str']"):
                    if child.text!=None:
                        desc += child.text
                ann_text+=desc
            # print(ann_text)
            song_res.append(ann_text)
        except:
            print('Song description retrieval failed')
            print(str(song_id)+', ', end='') # print(song,'\n',json_obj['response']['hits'], "\n\n\n")
            fail_count+=1
            if len(song_res)<8:
                for kk in range(len(song_res),8):
                    song_res.append(None)

    writer.writerow(song_res)

    idx+=1
    if idx%lim==0:
        file.close()

        results = pd.read_csv(str(start_idx)+'.csv')
        print(results.shape)
        # display(results.head())

        print('Starting ', idx)
        start_idx = idx
        file = open(str(idx)+'.csv', 'w')
        writer = csv.writer(file)
        writer.writerow(['song','artist','query_id','spotify_id','about','artist_speak','trivia','annotation'])


file.close()
results = pd.read_csv(str(start_idx)+'.csv')
print(results.shape)
# display(results.head())
