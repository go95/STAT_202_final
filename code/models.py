from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.base.datetools import dates_from_str


class BaselineModel(BaseEstimator, RegressorMixin):
    """
    Simple baseline model. Makes a mean constant prediction for each symbol
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.predictions = np.mean(y, axis=0)
        return self

    def predict(self, X):
        return np.repeat(self.predictions[np.newaxis, :], len(X), axis=0)


class SeperateModels(BaseEstimator, RegressorMixin):
    """
    A model, which combines two models for Period 1 and Period 2
    """
    def __init__(self, estimator1, estimator2, midpoint):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.midpoint = midpoint

    def fit(self, X, y):
        self.estimator1.fit(X, y)
        self.estimator2.fit(X, y)
        return self

    def predict(self, X):
        begin = self.estimator1.predict(X)
        end = self.estimator2.predict(X)

        result = begin
        result[self.midpoint:] = end[self.midpoint:]
        return result


class LastValueModel(BaseEstimator, RegressorMixin):
    """
    Simple baseline model. Makes a mean constant prediction for each symbol
    """
    def __init__(self, lag_order=1):
        self.lag_order=lag_order

    def fit(self, X, y):
        self.predictions = np.mean(y[-self.lag_order:], axis=0)
        return self

    def predict(self, X):
        return np.repeat(self.predictions[np.newaxis, :], len(X), axis=0)


class LastDayModel(BaseEstimator, RegressorMixin):
    """
    Simple baseline model. Repeats the previous day for the prediction
    """
    def __init__(self, close_slice, open_slice, timestamps_a_day=5040):
        self.close_slice = close_slice
        self.open_slice = open_slice
        self.timestamps_a_day = timestamps_a_day

    def fit(self, X, y):
        self.last_day = y[-self.timestamps_a_day:] - y[-self.timestamps_a_day - 1]
        self.prediction_factor = y[-1]
        self.prediction_factor[self.open_slice] = y[-1, self.close_slice]
        return self

    def predict(self, X):
        result = []
        for i in range((len(X) // self.timestamps_a_day) + 1):
            result.append(self.prediction_factor + self.last_day)
            self.prediction_factor = result[-1][-1]
        return np.concatenate(result, axis=0)[:len(X)]


class LastCloseModel(BaseEstimator, RegressorMixin):
    """
    Simple baseline model. Makes a mean constant prediction for each symbol based on the close value
    """
    def __init__(self, close_slice, open_slice):
        self.close_slice = close_slice
        self.open_slice = open_slice

    def fit(self, X, y):
        self.predictions = y[-1]
        self.predictions[self.open_slice] = y[-1, self.close_slice]
        return self

    def predict(self, X):
        return np.repeat(self.predictions[np.newaxis, :], len(X), axis=0)


class LinearNaiveModel(BaseEstimator, RegressorMixin): # fail to fallback really fast (need dynamic window if error in last is large??, but not too sensitive)! what to do? # + inday things! # robust to outliers. Fit to last value
    """ 
    Linear model, which takes the two points of the data and plots a line through them
    """
    def __init__(self, lag, clip = 100000000):
        self.lag = lag
        self.clip = clip

    def fit(self, X, y):
        last = y[-1, :]
        first = y[-self.lag, :]
        
        self.slope = (last - first) / self.lag
        self.init_prediction = y[-1, :]
        
        return self # fit on residuals? force last?

    def predict(self, X):
        trend = np.minimum(np.arange(len(X)) + 1, self.clip)[:, np.newaxis]
        predictions = trend * self.slope + self.init_prediction
        return predictions


class LinearClippedModel(BaseEstimator, RegressorMixin): # fail to fallback really fast (need dynamic window if error in last is large??, but not too sensitive)! what to do? # + inday things! # robust to outliers. Fit to last value
    """
    Linear OLS model on a subset of most recent datapoints
    """
    def __init__(self, lag, clip = 100000000):
        self.lag = lag
        self.clip = clip

    def fit(self, X, y):
        self.model = LinearRegression().fit(np.arange(self.lag)[:, np.newaxis], y[-self.lag:])
        
        
        return self # fit on residuals? force last?

    def predict(self, X):
        predictions = self.model.predict(
            np.minimum(np.arange(self.lag, self.lag + len(X))[:, np.newaxis]
                       , self.clip + self.lag))
        return predictions


class VarModel(BaseEstimator, RegressorMixin):
    """
    VAR model
    """
    
    def __init__(self, n_lags, forget=2 ** 31):
        self.n_lags = n_lags
        self.forget = forget
    
    def fit(self, X, y): # treat X as exog!!! and y as endog
        exog = X[-self.forget:]
        endog = y[-self.forget:]
        
        self.model = VAR(endog, exog)
        self.result = self.model.fit(self.n_lags)
        self.warm_start = endog[-self.n_lags:]
        return self
        
    def predict(self, X):
        return self.result.forecast(self.warm_start, len(X), X)


class DiffVarModel(BaseEstimator, RegressorMixin):
    """
    VAR model on the differences
    """
    
    def __init__(self, n_lags, forget=2 ** 31):
        self.n_lags = n_lags
        self.forget = forget
    
    def fit(self, X, y): # treat X as exog!!! and y as endog
        exog = np.diff(X[-self.forget:], axis=0)
        endog = np.diff(y[-self.forget:], axis=0)
        
        self.model = VAR(endog, exog)
        self.result = self.model.fit(self.n_lags)
        self.warm_start = endog[-self.n_lags:]
        self.baseline_y = y[-1:]
        self.baseline_X = X[-1:]
        return self
        
    def predict(self, X):
        exog = np.diff(np.concatenate([self.baseline_X, X]), axis=0)
        return self.baseline_y + np.cumsum(self.result.forecast(self.warm_start, len(X), exog), axis=0)


# +
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np


class IndexedTimeSeriesSplit(object):
    """
    Splits the sample given an index of a time variable and a set of thresholds for this variable
    """
    def __init__(self, day_th, day_idx, test_size=1):
        self.day_th = day_th
        self.day_idx = day_idx
        self.test_size = test_size
    
    def get_n_splits(self, X=None, y=None, groups=None):        
        return len(self.day_th)
        
    def split(self, X, y=None, groups=None):
        day = X[:, self.day_idx]
        day = day.astype(int)
        
        for th in self.day_th:
            yield np.where(day < th)[0], np.where((th <= day) & (day < th + self.test_size))[0]




# sklearn.model_selection.cross_val_predict(estimator, X, y=None
