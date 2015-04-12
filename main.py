print "Importing.."

import sys
reload(sys)
sys.setdefaultencoding('Windows-1251')
default_encoding = sys.getdefaultencoding()
import pandas as pd
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

TRAINING_SET_URL = "twitter_train.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=1, names=["user_id", "class"])
print df_users.head()
df_users = df_users[:3000]

print "Authentication.."

CONSUMER_KEY = "YPp9d9k4jOo8JUUbQVrcOBqYV"
CONSUMER_SECRET = "FXQngW71IOOgVs6mCpaMbEr0rqaEiUDHyERBnlwj5mQUKh9Jum"

ACCESS_TOKEN_KEY = "3064713329-LSUEMzqbfHlsJbj0dYLdxFsA9PYJGoyKwEOSpjn"
ACCESS_TOKEN_SECRET = "qXVKSE0rXlm0nyxkQTRPi3RhzbPzQzMQDKI27ZB9pOcEe"

api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN_KEY,
                  access_token_secret=ACCESS_TOKEN_SECRET)

print "Collecting data.."

def get_user_tweets(user_id):
    """returns list of tweets as dicts"""
    time = api.GetSleepTime("statuses/user_timeline")
    statuses = api.GetUserTimeline()
    dictionaries = []
    for i in range(0,len(statuses)):
        dictionaries.append(statuses[i].AsDict())
    return [{'lang': u'en',
             'favorited': False,
             'truncated': False,
             'text': u"So now I'm on the floor tweeting about it PROB w a black eye n swollen nose",
             'created_at': u'Mon Apr 06 05:59:50 +0000 2015',
             'retweeted': False,
             'source': u'<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>',
             'user': {'id': 984121344},
             'id': 584958674528964608}]

def get_user_records(df):
    user_records = []
    for i in range(0, df.size/2/100):
        print i, df.at[i, "user_id"]
        #try:
        users_list = []
        for j in range(0,99):
            if i*100+j >= df.size:
                break
            users_list.append(df.at[i*100+j,"user_id"])

        user = api.UsersLookup(users_list)
            #r = requests.get("https://api.twitter.com/1.1/users/lookup.json?user_id=601849857")
            #user = api.GetUser(df.at[i, "user_id"])
        #except:
            #print i, " Some exeption"
            #continue
        for j in range(0,99):
            if j >= len(user):
                break
            record = twitter_user_to_dataframe_record(user[j])
            if record.has_key('lat') and record.has_key('lng'):
                if record['lat'] is not None and record['lng'] is not None:
                    print record['lat'], '  ', record['lng']
                print i*100 +j, ' Added'
            user_records.append(record)
    return user_records


user_records = get_user_records(df_users)