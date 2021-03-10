from sklearn.model_selection import train_test_split
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from iterreg.ell1 import dual_primal


class BasisPursuitIterReg(LinearModel):
    def __init__(self, train_ratio=0.8, f_test=1, max_iter=1000, memory=20,
                 step=1, verbose=False):
        self.train_ratio = train_ratio
        self.f_test = f_test
        self.verbose = verbose
        self.max_iter = max_iter
        self.memory = memory
        self.step = step

    def fit(self, X, y, train_idx=None, test_idx=None):
        if train_idx is None or test_idx is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, shuffle=True, train_size=self.train_ratio)
        else:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

        def callback(w):
            return mean_squared_error(y_test, X_test @ w)

        w, thetas, mses = dual_primal(
            X_train, y_train, step=self.step, max_iter=self.max_iter,
            f_store=self.f_test, callback=callback, memory=self.memory,
            ret_all=False, verbose=self.verbose)

        self.coef_ = w
        self.mses = mses
        return self

    def debias(self, X, y):
        try:
            supp = (self.coef_ != 0)
            X_supp = X[:, supp]
            least_squares = LinearRegression(
                fit_intercept=False).fit(X_supp, y)
            self.coef_biased = self.coef_.copy()
            self.coef_ = least_squares.coef_.copy()
        except AttributeError:
            raise AttributeError("Model has not been fitted yet.")
