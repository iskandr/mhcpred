
import numpy as np 
from numpy import exp
import sklearn.ensemble
import sklearn.linear_model

class SelectiveRegressor(object):

    def __init__(self, cutoff, n_trees = 25):
        self.n_trees = n_trees
        self.cutoff = cutoff 
    
    def fit(self,X,Y,W=None):
        n = len(X)
        self.filters = []
        mask = Y <= self.cutoff 
        self.gate = sklearn.ensemble.RandomForestClassifier(n_estimators = self.n_trees)
        self.gate.fit(X, mask)
        self.regressor = sklearn.ensemble.RandomForestRegressor(n_estimators = self.n_trees)
        self.regressor.fit(X[mask], Y[mask])
        return self


    def predict(self, X):
        n = len(X)
        indices = np.arange(n)
        mask = self.gate.predict(X)
        Y = self.regressor.predict(X[mask])
        full = np.ones(n, dtype=float) * self.cutoff 
        full[indices] = Y
        return full

