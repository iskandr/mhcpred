import numpy as np 
import sklearn.linear_model


class LogLinearRegression(sklearn.linear_model.Ridge):
    def fit(self, X, Y, sample_weight = None):
        self._max_value = np.max(Y)
        Y = np.log(Y)
        return sklearn.linear_model.Ridge.fit(self, X, Y)


    def predict(self, X):
        transformed_Y = sklearn.linear_model.Ridge.predict(self, X)
        raw_values = np.exp(transformed_Y)
        return np.minimum(self._max_value, raw_values)