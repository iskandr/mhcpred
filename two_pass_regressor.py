
import numpy as np 
import sklearn.ensemble
import sklearn.linear_model

class TwoPassRegressor(object):

    def fit(self,X,Y,W=None):
        category_base = 100
        categories =  np.maximum(0, (np.log10(Y) / np.log10(category_base)).astype('int')) 
        self.first_pass = sklearn.ensemble.RandomForestClassifier(n_estimators = 20) #sklearn.linear_model.LogisticRegression()
        self.first_pass.fit(X, categories)
        
        Y = np.log(Y)
        self.regressors = [None] * (np.max(categories) + 1)
        for category in np.unique(categories):
            mask = categories == category
            print "-- Category #%d (base %d): %d samples" % (category, category_base, mask.sum())
            regressor = sklearn.linear_model.RidgeCV()
            regressor.fit(X[mask], Y[mask])
            self.regressors[category] = regressor
        return self


    def predict(self, X):
        probs = self.first_pass.predict_proba(X)
        combined = np.zeros(X.shape[0], dtype=float)
        weights = np.zeros_like(combined)

        for category_idx in xrange(probs.shape[1]):
            pred = self.regressors[category_idx].predict(X)
            prob = probs[:, category_idx] 
            combined += prob * pred 
        return np.exp(combined) 
