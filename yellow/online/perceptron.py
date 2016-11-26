import sys
from ..feature_map import RandomFourierSampler
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem

"""
Binary classification using Fourier Online Gradient Descent
@article{JMLR:v17:14-148,
  author  = {Jing Lu and Steven C.H. Hoi and Jialei Wang and Peilin Zhao and Zhi-Yong Liu},
  title   = {Large Scale Online Kernel Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2016},
  volume  = {17},
  number  = {47},
  pages   = {1-43},
  url     = {http://jmlr.org/papers/v17/14-148.html}
}
"""

class FOGDClassifier(object):
    def __init__(self, n_components=100, n_iter=1):
        self.rfs = RandomFourierSampler(n_components)
        self.clf = SGDClassifier(loss='hinge', penalty='l2', shuffle=True, n_iter=n_iter)
        self.count= 0

    def fit(self, X, y):
        if self.count == 0:
            X_tran = self.rfs.fit_transform(X)
        else:
            X_tran = self.rfs.transform(X)
        self.count += 1
        self.clf.fit(X_tran, y)

    def partial_fit(self, X, y):
        if self.count == 0:
            X_tran = self.rfs.fit_transform(X)
        else:
            X_tran = self.rfs.transform(X)
        self.count += 1
        self.clf.partial_fit(X_tran, y)

    def predict(self, X):
        X_tran = self.rfs.transform(X)
        y_pred = self.clf.predict(X_tran)
        return y_pred

class NOGDClassifier(object):
    def __init__(self, n_components=100, n_iter=1):
        self.nys = Nystroem(n_components=n_components)
        self.clf = SGDClassifier(loss='hinge', penalty='l2', shuffle=True, n_iter=n_iter)
        self.count= 0

    def fit(self, X, y):
        if self.count == 0:
            X_tran = self.nys.fit_transform(X)
        else:
            X_tran = self.nys.transform(X)
        self.count += 1
        self.clf.fit(X_tran, y)

    def partial_fit(self, X, y):
        if self.count == 0:
            X_tran = self.nys.fit_transform(X)
        else:
            X_tran = self.nys.transform(X)
        self.count += 1
        self.clf.partial_fit(X_tran, y)

    def predict(self, X):
        X_tran = self.nys.transform(X)
        y_pred = self.clf.predict(X_tran)
        return y_pred

