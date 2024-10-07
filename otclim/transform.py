import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


class QuantileMinMaxScaler(BaseEstimator, TransformerMixin):
    """Transform data by scaling the specified quantiles to [0, 1]."""

    def __init__(self, qmin: float, qmax: float):
        self.qmin = qmin
        self.qmax = qmax

    def fit(self, X, y=None):
        X_qmin, X_qmax = np.quantile(X, q=[self.qmin, self.qmax])
        self.X_qmin_ = X_qmin
        self.X_qmax_ = X_qmax
        self.X_range_ = X_qmax - X_qmin
        return self

    def transform(self, X):
        return (X - self.X_qmin_) / self.X_range_

    def inverse_transform(self, X):
        return (X * self.X_range_) + self.X_qmin_

    def set_output(self, *, transform=None): ...


def power_10(x):
    return 10**x


def get_log10_tf():
    """Return a transformer that applies log10 transformation."""
    return FunctionTransformer(func=np.log10, inverse_func=power_10, validate=True)


def get_log_qmms_tf():
    """Return a pipeline that applies log10 transformation and quantile min-max scaling."""
    return make_pipeline(get_log10_tf(), QuantileMinMaxScaler(qmin=0.25, qmax=0.75))
