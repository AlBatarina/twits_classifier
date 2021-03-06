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
import twitter
import requests
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import DictVectorizer
import nltk
import random as rnd

def get_user_tweets(user_id):
    """returns list of tweets as dicts"""
    while True:
        try:
            statuses = api.GetUserTimeline(user_id, count=200)
            break
        except:
            waitTime = api.GetSleepTime("statuses/user_timeline")
            if waitTime == 0:
                return None
            print "Waiting ", waitTime, "seconds"
            time.sleep(waitTime)
    dictionaries = []
    for i in range(0,len(statuses)):
        dict = statuses[i].AsDict() #statuses[i].GetRetweeted()
        if  dict.has_key('retweeted_status') or statuses[i].GetInReplyToUserId() != None \
                or dict['lang'] != 'en' or statuses[i].GetInReplyToStatusId() != None\
                or dict['text'][0:3] == "RT ":
            continue
        for key in dict.keys():
            if key not in ['text', 'created_at', 'id']:
                dict.pop(key)
        dictionaries.append(dict)
    if len(dictionaries) == 0:
        return None
    return dictionaries

def get_words(text):
    """returns list of words"""
    refs = re.findall('http\S*', text)
    refs.extend(re.findall('#\S*', text))
    refs.extend(re.findall('@\S*', text))
    refs.extend(re.findall('\s\d+', text))
    refs.extend(re.findall('\d+\s', text))
    for ref in refs:
        text = text.replace(ref,'')
    tokenizer = RegexpTokenizer('\w+')#|[^\w\s]+')
    return tokenizer.tokenize(text)

def get_tokens(words):
    """returns list of tokens"""
    wnl = WordNetLemmatizer()
    for i in range(0, len(words)):
        words[i] = words[i].lower()
        words[i] = re.sub(ur"\W", "", words[i], flags=re.U)
        wnl.lemmatize(words[i])
    stpwrd = stopwords.words('english')
    stpwrd.extend(['m','re','o','d','vs','w','3','2','rt','u','ll','ve'])
    tokens = [i for i in words if i not in stpwrd]
    #print tokens
    return tokens

def get_tweet_tokens(tweet):
    return get_tokens(get_words(tweet))

def collect_users_tokens(df_users):
    """returns users list and list of user dicts. Each dict contains frequence of user tokens"""
    '''
    print "Collecting data.."
    users_tweets = []
    users = []
    for i in range(0, len(df_users)):
        print i, df_users.at[i, 'class']
        dictList = get_user_tweets(df_users.at[i, 'user_id'])
        if dictList is not None:
            users.append(df_users.at[i, 'user_id'])
            users_tweets.append(dictList)

    print "Saving dictionaries.."
    tweets = open("./tweets_short", 'w')
    pck.dump(users_tweets,tweets)
    tweets.close()
    usrs = open("./users", 'w')
    pck.dump(users,usrs)
    usrs.close()
    '''
    print "Opening dictionaries.."
    #raw_input()
    tweets = open("./tweets", 'r')
    users_tweets = pck.load(tweets)
    tweets.close()
    usrs = open("./users", 'r')
    users = pck.load(usrs)
    usrs.close()

    users_tokens = []
    #nltk.download()
    for i in range(0, len(users_tweets)):
        tokens = []
        for j in range(0, len(users_tweets[i])):
            #print users_tweets[i][j]
            tokens.extend(get_tweet_tokens(users_tweets[i][j]['text']))
        tokenDict = dict((x, tokens.count(x)) for x in set(tokens))
        users_tokens.append(tokenDict)
    return users, users_tokens

TRAINING_SET_URL = "twitter_train.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=1, names=["user_id", "class"])
print df_users.head()

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
vs = v.fit_transform(users_tokens)

np.savez("TokensMatrix", data=vs, users=users, users_tokens=users_tokens )

def draw_tag_cloud(v, vs):
    """Draws tag cloud of found tokens"""
    tagsNumb = 100
    tokenFrequency = np.sum(vs.toarray(), axis=0)
    print tokenFrequency
    indexPopular = np.argsort(tokenFrequency)[len(tokenFrequency)-tagsNumb:]
    tokenPopular = np.take(tokenFrequency, indexPopular)
    print tokenPopular
    maxFreq = np.max(tokenPopular)
    minFreq = np.min(tokenPopular)
    print maxFreq, minFreq
    pl.figure(figsize=(30,30))
    rnd.seed()
    colors = ['b','g','r','k','y','c','m']
    for i in range(len(indexPopular)-1, -1, -1):
        print i
        size = 30/((maxFreq-tokenFrequency[indexPopular[i]])/(maxFreq-minFreq)+1)
        n = 295
        alpha = n*np.pi/tagsNumb*(tagsNumb-i)
        r = np.log(np.log(float(tagsNumb-i)/tagsNumb/2 + 1) + 1)*0.9 + 0.14
        print 'alpha', alpha
        pl.text(0.46+r*np.cos(alpha), 0.5+r*np.sin(alpha), v.get_feature_names()[indexPopular[i]],
                {'color' : colors[i%len(colors)], 'fontsize' : size} )
    pl.show()
    return

draw_tag_cloud(v,vs)
