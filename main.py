print "Importing.."

import sys
reload(sys)
sys.setdefaultencoding('Windows-1251')
default_encoding = sys.getdefaultencoding()
import pandas as pd
import time
import cPickle as pck
import numpy as np
import pylab as pl
import mpl_toolkits.basemap as bm
import twitter
import requests
import datetime
import dateutil
import csv
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction import DictVectorizer
import nltk

def get_user_tweets(user_id):
    """returns list of tweets as dicts"""
    waitTime = api.GetSleepTime("statuses/user_timeline")
    if waitTime != 0:
        print "Waiting ", waitTime, "seconds"
    time.sleep(waitTime)
    try:
        statuses = api.GetUserTimeline(user_id, count=200)
    except:
        return None
    print len(statuses)
    dictionaries = []
    for i in range(0,len(statuses)):
        dict = statuses[i].AsDict()
        if statuses[i].GetRetweeted() or statuses[i].GetInReplyToUserId() != None or dict['lang'] != 'en':
            continue
        for key in dict.keys():
            if key != 'text':
                dict.pop(key)
        dictionaries.append(dict)
    if len(dict) == 0:
        return None
    return dictionaries

def get_words(text):
    """returns list of words"""
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer('\w+|[^\w\s]+')
    return tokenizer.tokenize(text)

def get_tokens(words):
    """returns list of tokens"""
    wnl = WordNetLemmatizer()
    for i in range(0, len(words)):
        words[i] = words[i].lower()
        words[i] = re.sub(ur"\W", "", words[i], flags=re.U)
        wnl.lemmatize(words[i])
    tokens = [i for i in words if i not in stopwords.words('english')]
    return tokens

def get_tweet_tokens(tweet):
    return get_tokens(get_words(tweet))

def collect_users_tokens(df_users):
    """returns users list and list of user dicts. Each dict contains frequence of user tokens"""
    print "Collecting data.."
    users_tweets = []
    users = []
    for i in range(0, len(df_users)):
        print i, df_users.at[i, 'class']
        dictList = get_user_tweets(df_users.at[i, 'user_id'])
        if dictList is not None:
            users.append(df_users.at[i, 'user_id'])
            users_tweets.append(dictList)
    print users_tweets[0]
    '''
    print "Saving dictionaries.."
    tweets = open("./tweets", 'w')
    pck.dump(users_tweets,tweets)
    tweets.close()

    print "Opening dictionaries.."
    raw_input()
    tweets = open("./tweets", 'r')
    users_tweets = pck.load(tweets)
    tweets.close()
    '''
    users_tokens = []
    nltk.download()
    for i in range(0, len(users_tweets)):
        users_tokens.append(get_tweet_tokens(users_tweets[i]['text']))
    return users, users_tokens

TRAINING_SET_URL = "twitter_train.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=1, names=["user_id", "class"])
print df_users.head()
df_users = df_users[:10]

print "Authentication.."

CONSUMER_KEY = "YPp9d9k4jOo8JUUbQVrcOBqYV"
CONSUMER_SECRET = "FXQngW71IOOgVs6mCpaMbEr0rqaEiUDHyERBnlwj5mQUKh9Jum"

ACCESS_TOKEN_KEY = "3064713329-LSUEMzqbfHlsJbj0dYLdxFsA9PYJGoyKwEOSpjn"
ACCESS_TOKEN_SECRET = "qXVKSE0rXlm0nyxkQTRPi3RhzbPzQzMQDKI27ZB9pOcEe"

api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN_KEY,
                  access_token_secret=ACCESS_TOKEN_SECRET)

users, users_tokens = collect_users_tokens(df_users)
print "Transforming to matrix.."
v = DictVectorizer()
#vs = v.fit_transform(users_tokens)

