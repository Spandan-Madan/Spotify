from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser
import pandas as pd
import pprint
import matplotlib.pyplot as pd

DEVELOPER_KEY = '' # GET A DEVELOPER KEY FOR YT API
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def youtube_search(q, max_results=3,order="relevance", token=None, location=None, location_radius=None):

    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
    q=q,
    type="video",
    pageToken=token,
    order = order,
    part="id,snippet", # Part signifies the different types of data you want
    maxResults=max_results,
    location=location,
    locationRadius=location_radius).execute()

    # print(search_response)

    title = []
    channelId = []
    channelTitle = []
    categoryId = []
    videoId = []
    viewCount = []
    likeCount = []
    dislikeCount = []
    commentCount = []
    favoriteCount = []
    category = []
    tags = []
    videos = []
    comments = []

    for search_result in search_response.get("items", [])[:3]:
        if search_result["id"]["kind"] == "youtube#video":

            title.append(search_result['snippet']['title'])

            videoId.append(search_result['id']['videoId'])

            response = youtube.videos().list(
            part='statistics, snippet',
            id=search_result['id']['videoId']).execute()

            channelId.append(response['items'][0]['snippet']['channelId'])
            channelTitle.append(response['items'][0]['snippet']['channelTitle'])
            categoryId.append(response['items'][0]['snippet']['categoryId'])
            favoriteCount.append(response['items'][0]['statistics']['favoriteCount'])
            viewCount.append(response['items'][0]['statistics']['viewCount'])
            likeCount.append(response['items'][0]['statistics']['likeCount'])
            dislikeCount.append(response['items'][0]['statistics']['dislikeCount'])

            if 'commentCount' in response['items'][0]['statistics'].keys():
                commentCount.append(response['items'][0]['statistics']['commentCount'])
            else:
                commentCount.append([None])

                if 'tags' in response['items'][0]['snippet'].keys():
                    tags.append(response['items'][0]['snippet']['tags'])
                else:
                    tags.append([None])

            # Getting first 200 comments
            comment_response = youtube.commentThreads().list(
            part="snippet",
            videoId=search_result['id']['videoId'],
            textFormat="plainText",
            maxResults=100).execute()

            comment_list = []
            for item in comment_response["items"]:
                comment = item["snippet"]["topLevelComment"] # author = comment["snippet"]["authorDisplayName"]
                text = comment["snippet"]["textDisplay"]
                comment_list.append(text)
            comments.append(comment_list)

    youtube_dict = {'tags':tags,'channelId': channelId,'channelTitle': channelTitle,'categoryId':categoryId,'title':title,'videoId':videoId,'viewCount':viewCount,'likeCount':likeCount,'dislikeCount':dislikeCount,'commentCount':commentCount,'favoriteCount':favoriteCount, 'comments': comments}

    return youtube_dict
