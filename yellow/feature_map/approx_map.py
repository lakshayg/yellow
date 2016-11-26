import numpy as np
import warnings

# approximate feature map for RBF kernel using
# random fourier features
class RandomFourierSampler(object):
    def __init__(self, n_components=100):
        self.W = None
        self.n_components = n_components
        if n_components % 2 != 0:
            warnings.warn('n_components must be an even number for Random Fourier Features')

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        self.W = np.random.normal(0, 1, [n_features, self.n_components/2])
        return self

    def fit_transform(self, X, y=None):
        n_samples, n_features = X.shape
        self.W = np.random.normal(0, 1, [n_features, self.n_components/2])
        M = np.dot(X, self.W)
        X_new = np.hstack([np.cos(M), np.sin(M)])/np.sqrt(self.n_components/2)
        return X_new

    def get_params(self):
        return {'weights':self.W}

    def transform(self, X, y=None):
        if self.W is None:
            raise "fit or fit_transform must be called before transform can be used"
        M = np.dot(X, self.W)
        X_new = np.hstack([np.cos(M), np.sin(M)])/np.sqrt(self.n_components/2)
        return X_new

