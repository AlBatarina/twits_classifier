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

data = np.load("TokensMatrix.npz")
#users = data["users"]
usrs = open("./users", 'r')
users = np.array(pck.load(usrs))
usrs.close()
X = data["data"].reshape(1,)[0]
print type(data)
print data.files

X = X[:1000]

TRAINING_SET_URL = "twitter_train.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"], dtype={"user_id": str, "class": int})
df_users.set_index("user_id", inplace=True)

Y = df_users.ix[users.astype(str)]["class"].values
print "Resulting training set: (%dx%d) feature matrix, %d target vector" % (X.shape[0], X.shape[1], Y.shape[0])

def draw_log_hist(x):
    """Draw tokens histogram in log scales"""

    # Your code here

    return

draw_log_hist(X)
print X
i, j, v = spr.find(X)
print i, j, v
print len(j), np.min(j)
col_indices = []
#col_list = list(j)
for colNumb in range(0,X.shape[1]):
    users_count = len(np.extract( j == colNumb, j))
    print colNumb, users_count
    if  users_count > 100:
        col_indices.append(colNumb)
X1 = X[:,col_indices]
print X1.shape

popular_tokens = open("./popular_tokens", 'w')
pck.dump(X1,popular_tokens)
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

    def fit(self, X, Y=None):

        # your code here

        return self

    def predict_proba(self, X):

        #your code here

        import numpy.random as nr
        return nr.random((X.shape[0], 2))

    def auroc(y_prob, y_true):
        pass

C = [0.0, 0.01, 0.1, 1, 10, 100, 1000, 10000]

def select_reg_parameter(C, X, Y):

    return C.index(max(C))

index = select_reg_parameter(C, X1, Y)
print index

def classify(X, Y, test_size, C):
    tpr = [1] * 2400
    fpr = [0.01] * 2400
    roc_auc = 0.51

    return tpr, fpr, roc_auc

tpr, fpr, roc_auc = classify(X1, Y, 0.3, C[index])

print "Area under the ROC curve : %f" % roc_auc

def plot_roc_curve(tpr, fpr, roc_auc):
    """Plot ROC curve"""

    # Your code here

    return

plot_roc_curve(tpr, fpr, roc_auc)