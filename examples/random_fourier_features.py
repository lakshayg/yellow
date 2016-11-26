from pydata import *
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from time import time
import numpy as np
from sklearn import svm

X, y, _ = load_mnist()
X, y = random_sample(X, y, 8000)
pca = PCA(n_components=750)
X = pca.fit_transform(X)
X = norm_zscore(X)
y = np.ravel(y)

print 'dataset:', X.shape

T1 = []
T2 = []
Acc = []

def print_stats(name, t1, t2, t3, y_test, y_pred):
    total_len = 30
    d1 = "="*(total_len/2 - len(name)/2)
    d2 = "="*(total_len/2 - len(name) + len(name)/2)
    print "{}[ {} ]{}".format(d1,name,d2)
    print "Train time: %5.4f" % (t2-t1)
    print "Test time : %5.4f" % (t3-t2)
    print "Accuracy  : %3.2f" % (accuracy_score(y_test, y_pred))
    print ""
    T1.append(t2-t1)
    T2.append(t3-t2)
    Acc.append(accuracy_score(y_test, y_pred))

try:
    # train a rbf-svm
    if 0:
        T1 = []
        T2 = []
        Acc = []
        for i in range(20):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            clf = svm.SVC(kernel='rbf', cache_size=1)
            t1 = time(); clf.fit(X_train, y_train)
            t2 = time(); y_pred = clf.predict(X_test)
            print_stats('RBF-SVM {}'.format(i), t1, t2, time(), y_test, y_pred)
            del clf
        print "mean train time:", np.mean(T1)
        print "mean test  time:", np.mean(T2)
        print "mean accuracy  :", np.mean(Acc)

    # train a rff + linear-svm
    if 0:
        T1 = []
        T2 = []
        Acc = []
        for i in range(20):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            from yellow.feature_map import RandomFourierSampler
            sampler = RandomFourierSampler(n_components=500)
            clf = svm.LinearSVC()
            t1 = time(); X_train_new = sampler.fit_transform(X_train); clf.fit(X_train_new, y_train)
            t2 = time(); X_test_new = sampler.transform(X_test); y_pred = clf.predict(X_test_new)
            print_stats('RFF-SVM {}'.format(i), t1, t2, time(), y_test, y_pred)
        print "mean train time:", np.mean(T1)
        print "mean test  time:", np.mean(T2)
        print "mean accuracy  :", np.mean(Acc)

    # train a linear-svm
    if 1:
        T1 = []
        T2 = []
        Acc = []
        for i in range(20):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            clf = svm.LinearSVC()
            t1 = time(); clf.fit(X_train, y_train)
            t2 = time(); y_pred = clf.predict(X_test)
            print_stats('Linear-SVM', t1, t2, time(), y_test, y_pred)
        print "mean train time:", np.mean(T1)
        print "mean test  time:", np.mean(T2)
        print "mean accuracy  :", np.mean(Acc)

except KeyboardInterrupt:
        print "mean train time:", np.mean(T1)
        print "mean test  time:", np.mean(T2)
        print "mean accuracy  :", np.mean(Acc)

