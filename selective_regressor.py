
import numpy as np 
from numpy import exp
import sklearn.ensemble
import sklearn.linear_model

class SelectiveRegressor(object):

    def __init__(self, n_trees = 25, cutoff = 4000, n_filters = 3):
        self.n_trees = n_trees
        self.cutoff = cutoff 
        self.n_filters = n_filters

    def fit(self,X,Y,W=None):
        n = len(X)
        cutoff = self.cutoff
        self.filters = []

        for _ in xrange(self.n_filters):
            mask = Y < cutoff 
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators = self.n_trees)
            clf.fit(X, mask)
            self.filters.append(clf)
            X = X[mask]
            Y = Y[mask]
            cutoff /= 2

        self.regressor = sklearn.ensemble.RandomForestRegressor(n_estimators = self.n_trees)

        self.regressor.fit(X, np.log(Y))
        log_Ypred = self.regressor.predict(X)
        Ypred = np.exp(log_Ypred)
        training_error = np.median(np.abs(Ypred - Y))

        print "Training error after filtering: %0.4f (%d/%d samples)" % (training_error, len(Ypred), n)
        return self


    def predict(self, X):
        n = len(X)
        indices = np.arange(n)
        for clf in self.filters:
            mask = clf.predict(X)
            indices = indices[mask]    
            X = X[mask]

        Y = self.regressor.predict(X)
        full_mask = np.zeros(n, dtype=bool)
        full_mask[indices] = True
        return full_mask, np.exp(Y)
