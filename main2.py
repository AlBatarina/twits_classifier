print "Importing.."

import sys
reload(sys)
sys.setdefaultencoding('Windows-1251')
default_encoding = sys.getdefaultencoding()
import random as pr
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import sklearn.cross_validation as cv
import sklearn.metrics as sm
import cPickle as pck
import scipy.sparse as spr
import matplotlib.axes as ax
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from numpy.linalg import norm
from math import exp

#import parser
#print parser.__file__

data = np.load("TokensMatrix.npz")
#users = data["users"]
usrs = open("./users", 'r')
users = np.array(pck.load(usrs))
usrs.close()
X = data["data"].reshape(1,)[0]

TRAINING_SET_URL = "twitter_train.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"], dtype={"user_id": str, "class": int})
df_users.set_index("user_id", inplace=True)

Y = df_users.ix[users.astype(str)]["class"].values
print "Resulting training set: (%dx%d) feature matrix, %d target vector" % (X.shape[0], X.shape[1], Y.shape[0])

def draw_log_hist(X):
    """Draw tokens histogram in log scales"""
    i, j, v = spr.find(X)
    tokens_count = np.zeros(X.shape[0])
    indexes = np.arange(0,X.shape[0])
    for colNumb in range(0,X.shape[1]):
        users_count = len(np.extract( j == colNumb, j))
        print colNumb, users_count
        if (users_count):
            tokens_count[users_count] += 1
    nonzero_ind = np.nonzero(tokens_count)
    x = np.take(indexes,nonzero_ind)
    y = np.take(tokens_count,nonzero_ind)
    print x
    print y
    pl.figure(figsize=(30,30))
    pl.title("Token frequency distribution")
    pl.xlabel('users_count', size = 12)
    pl.ylabel('tokens_count', size = 12)
    ax = pl.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    pl.scatter(x, y)
    pl.show()
    return

def select_popular(X, users_tokens, min_users_count = X.shape[0]/700):
    i, j, v = spr.find(X)
    print i, j, v
    print len(j), np.min(j)
    col_indices = []
    #col_list = list(j)
    for colNumb in range(0,X.shape[1]):
        users_count = len(np.extract( j == colNumb, j))
        print colNumb, users_count
        if  users_count > min_users_count:
            col_indices.append(colNumb)
    print col_indices
    return X[:,col_indices], users_tokens[col_indices]

#draw_log_hist(X[:500])

usrtck = open("./users_tokens", 'r')
users_tokens = pck.load(usrtck)
usrtck.close()
print users_tokens
users_tokens = np.array(users_tokens)

#X1, users_tokens = select_popular(X, users_tokens, 2)
'''
popular_tokens = open("./popular_tokens", 'w')
pck.dump(X1,popular_tokens)
popular_tokens.close()
'''
popular_tokens = open("./popular_tokens", 'r')
X1 = pck.load(popular_tokens)
popular_tokens.close()
print X1.shape

usrtck = open("./tokens_selected", 'w')
users_tokens = pck.dump(usrtck)
usrtck.close()

def find_correlated_features(X, Y, users_tokens = []):
    # replace this code to find really correlated features
    columns = []
    X1 = X.tocsc()
    for j in range(0,X.shape[1]):
        z = np.vstack((X1[:,j].toarray().reshape(7616,),Y))
        a = np.ma.compress_cols(np.ma.masked_invalid(np.float64(z)))
        if len(a) == 0:
            continue
        cov = np.cov(a)
        corr = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
        #print corr
        if abs(corr) >= 0.1:
            print "Correlated feature: %s -> %.2f" % (users_tokens[j], corr)
            columns.append(j)
    return X1[:,columns], users_tokens[columns]

X1, users_tokens = find_correlated_features(X1, Y, users_tokens)
print X1.shape
X1 = X1.toarray()
'''
usrtck = open("./tokens_selected", 'w')
pck.dump(users_tokens, usrtck)
usrtck.close()
'''
USER_NAME = "al.batarina"
OPTIMIZATION_ALGORITHMS = ["stochastic gradient descent", "Newton method"]
REGULARIZATIONS = ["L1", "L2"]

print "My homework 5 algorithm is: Logistic regression with %s regularization optimized by %s" % (
    REGULARIZATIONS[hash(USER_NAME) % 2],
    OPTIMIZATION_ALGORITHMS[hash(USER_NAME[::-1]) % 2]
)

class LogisticRegression():
    def __init__(self, C):
        self.C = C
        self.w = None

    def fit(self, X, Y=None):
        w0 = np.full(X.shape[1],1/X.shape[1])
        eps = 0.005
        self.w = self.grad_optimize(w0, X, Y, eps)
        return self

    def predict_proba(self, X):
        proba = np.dot(X,self.w)
        for i in range(0,proba.shape[0]):
            if proba[i] < 0:
                proba[i] = 0
            if proba[i] > 1:
                proba[i] = 1
        return proba
        #import numpy.random as nr
        #return nr.random(X.shape[0])

    def sigma(self, a):
        if a > 28:
            return 1
        return 1 - 1/float(1+exp(a))

    def grad_err(self, w, X, Y):
        result = 0
        for i in range(0,X.shape[0]):
            y = self.sigma(np.dot(w, X[i]))
            t = Y[i]
            result += np.dot(y-t, X[i])
        result += self.C*np.sign(w)
        return result

    def grad_optimize(self, w0, X, Y, eps):
        k = 30
        w = w0
        while True:
            k = k + 1
            print k
            delta = -1*self.grad_err(w, X, Y)/k/k/k
            w = w + delta
            #print 'w ', w
            print 'norm ',norm(delta, 2)
            if norm(delta, 2) < eps:
                break
        return w

def auroc(y_prob, y_true):
    #threshold = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.99, 0.9999, 0.99999, 0.999999, 1.01]
    threshold = np.sort(np.unique(y_prob))
    threshold[0] = 0
    threshold[threshold.shape[0]-1] = 1
    tpr = np.empty_like(threshold)
    fpr = np.empty_like(threshold)
    for i in range(0, len(threshold)):
        print threshold[i]
        predicted = np.empty_like(y_prob)
        for j in range(0,y_prob.shape[0]):
            if threshold[len(threshold)-1-i] != 1:
                predicted[j] = int(y_prob[j] >= threshold[len(threshold)-1-i])
            else:
                predicted[j] = 0
        a = np.extract(y_true == 1, predicted-y_true)
        tpr[i] = float(a.shape[0] - np.count_nonzero(a))/np.count_nonzero(y_true)
        b = np.extract(y_true == 0, predicted-y_true)
        fpr[i] = np.count_nonzero(b)/float(y_true.shape[0]-np.count_nonzero(y_true))
    #roc = interp1d(fpr, tpr, kind='linear')
    roc_auc = trapz(tpr, fpr)
    print 'tpr', tpr
    print 'fpr', fpr
    print 'roc_auc', roc_auc
    return tpr, fpr, roc_auc



C = [0.0, 0.01, 0.1, 1, 10, 100, 1000, 10000]

def select_reg_parameter(C, X, Y):
    #skf = cv.StratifiedKFold(X)
    roc_auc = np.empty_like(C)
    nsplit = 3
    train_size = X.shape[0]*(nsplit-1)/nsplit
    for i in range(2,8):
        LR = LogisticRegression(C[i])
        #for train_index, test_index in skf:
        LR.fit(X[:train_size],Y[:train_size])
        y_prob = LR.predict_proba(X[train_size:])
        print 'y_prob', y_prob
        tpr, fpr, roc_auc[i] = auroc(y_prob,Y[train_size:])
    return np.argmax(roc_auc)

index = 4
index = select_reg_parameter(C, X1, Y)
print 'Regularisation parameter is ', C[index]

def classify(X, Y, nsplit, c):
    train_size = X.shape[0]*(nsplit-1)/nsplit
    LR = LogisticRegression(c)
    LR.fit(X[:train_size],Y[:train_size])
    y_prob = LR.predict_proba(X[train_size:])
    print y_prob
    return auroc(y_prob, Y[train_size:])

def plot_roc_curve(tpr, fpr, roc_auc):
    """Plot ROC curve"""
    roc = interp1d(fpr, tpr, kind='linear')
    pl.figure(figsize=(30,30))
    pl.title("ROC curve")
    pl.xlabel('fpr', size = 12)
    pl.ylabel('tpr', size = 12)
    x = np.linspace(0, 1, 10)
    pl.plot(x, roc(x))
    pl.show()
    return


tpr, fpr, roc_auc = classify(X1, Y, 3, C[index])
print "Area under the ROC curve : %f" % roc_auc
plot_roc_curve(tpr, fpr, roc_auc)

TRAINING_SET_URL = "twitter_test.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"], dtype={"user_id": str, "class": str})
df_users.set_index("user_id", inplace=True)

data = np.load("TokensMatrixTest.npz")
usrs = open("./users_test", 'r')
users = np.array(pck.load(usrs))
usrs.close()
X = data["data"].reshape(1,)[0]
