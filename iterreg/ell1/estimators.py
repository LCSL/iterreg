import numpy as np
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from numpy.linalg import norm

from iterreg.utils import power_method, shrink


def _dual_primal_callback(
        X_train, y_train, X_test, y_test, max_iter=100, memory=100, f_test=1,
        verbose=False):
    """
    Do Pock-Cambolle iterations on train data and stop when MSE on test
    data stops decreasing.
    """
    best_mse = np.mean(y_test ** 2)
    mses = np.zeros(max_iter // f_test)
    best_w = np.zeros(X_train.shape[0])
    n_non_decrease = 0

    if sparse.issparse(X_train):
        tau = 1 / power_method(X_train)
    else:
        tau = 1 / norm(X_train, ord=2)
    sigma = tau
    w = np.zeros(X_train.shape[1])
    theta = np.zeros(X_train.shape[0])
    theta_old = np.zeros(X_train.shape[0])

    for k in range(max_iter):
        w = shrink(w - tau * X_train.T @ (2 * theta - theta_old), tau)
        theta_old[:] = theta
        theta += sigma * (X_train @ w - y_train)
        if k % f_test == 0:
            y_pred = X_test @ w
            mse = np.mean((y_pred - y_test) ** 2)
            mses[k // f_test] = mse
            if verbose:
                print("Iter %d, mse: %.3f" % (k, mse))
            if mse < best_mse:
                best_mse = mse
                best_w = w.copy()
                n_non_decrease = 0
            else:
                n_non_decrease += 1
            if n_non_decrease == memory:
                if verbose:
                    print("No improvement for %d MSE (best: %d), exit" %
                          (memory, k - memory))
                break
    else:
        print("Convergence warning: mse was still decreasing when "
              "max_iter was reached")

    return best_w, mses[:k // f_test + 1]


class BasisPursuitIterReg(LinearModel):
    def __init__(self, train_ratio=0.8, f_test=1, max_iter=1000, memory=20,
                 verbose=False):
        # __super__(Lasso, self).__init__()
        self.train_ratio = train_ratio
        self.f_test = f_test
        self.verbose = verbose
        self.max_iter = max_iter
        self.memory = memory

    def fit(self, X, y, train_idx=None, test_idx=None):
        if train_idx is None or test_idx is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, shuffle=True, train_size=self.train_ratio)
        else:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

        w, mses = _dual_primal_callback(
            X_train, y_train, X_test, y_test, self.max_iter, self.memory,
            self.f_test, self.verbose)

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
