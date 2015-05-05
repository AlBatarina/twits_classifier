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

data = np.load("TokensMatrix.npz")
#users = data["users"]
usrs = open("./users", 'r')
users = np.array(pck.load(usrs))
usrs.close()
X = data["data"].reshape(1,)[0]
print type(data)
print data.files

#X = X[:500]

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

def select_popular(X, min_users_count = X.shape[0]/700):
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
    return X[:,col_indices]

#draw_log_hist(X[:500])
#X1 = select_popular(X, 100)
#print X1.shape
'''
popular_tokens = open("./popular_tokens", 'w')
pck.dump(X1,popular_tokens)
popular_tokens.close()
'''
popular_tokens = open("./popular_tokens", 'r')
X1 = pck.load(popular_tokens)
popular_tokens.close()

X1 = X1.toarray()

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



        return self

    def predict_proba(self, X):
        if self.w == None:
            return None
        return self.w*X

def auroc(y_prob, y_true):
    threshold = np.linspace(0, 1, 10)
    for i in range(0,len(threshold)):
        predicted = np.empty_like(y_prob)
        for j in range(0,y_prob.shape[0]):
            if y_prob[j] >= threshold[i]:
                predicted[j] = 1
            else:
                predicted[j] = 0


def grad_err(a):
    pass

def gd(a0, epsilon):
    k = 0
    a = a0
    while True:
        k = k + 1
        delta = -1/k*grad_err(a)
        a = a + delta
        if np.norm(delta) < epsilon:
            break
    return a

C = [0.0, 0.01, 0.1, 1, 10, 100, 1000, 10000]

def select_reg_parameter(C, X, Y):
    skf = cv.cross_validation.StratifiedKFold(X)
    tpr = []
    fpr = []
    for c in C:
        LR = LogisticRegression(c)
        for train_index, test_index in skf:
            LR.fit(X[train_index],Y)
            classes = LR.predict(X[test_index])
            tpri, fpri = tprfpr(classes, Y)
            tpr.append(tpri)
            fpr.append(fpri)
    return C.index(max(C))

index = select_reg_parameter(C, X1, Y)
print index

def classify(X, Y, nsplit, c):
    train_size = X.shape[0]*(nsplit-1)/nsplit
    LR = LogisticRegression(c)
    LR.fit(X[:train_size],Y[:train_size])
    y_prob = LR.predict_proba(X[train_size:])
    return LR.auroc(y_prob, Y[train_size:])

def plot_roc_curve(tpr, fpr, roc_auc):
    """Plot ROC curve"""

    # Your code here

    return

TRAINING_SET_URL = "twitter_test.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"], dtype={"user_id": str, "class": str})
df_users.set_index("user_id", inplace=True)

data = np.load("TokensMatrixTest.npz")
usrs = open("./users_test", 'r')
users = np.array(pck.load(usrs))
usrs.close()
X = data["data"].reshape(1,)[0]

tpr, fpr, roc_auc = classify(X1, Y, 3, C[index])
print "Area under the ROC curve : %f" % roc_auc
plot_roc_curve(tpr, fpr, roc_auc)


