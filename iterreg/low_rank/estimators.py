import numpy as np

from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import LinearRegression

from iterreg.utils import shrink


def _dual_primal_lowrank_callback(
        mask_train, mask_test, Y,  max_iter=1000, memory=100,
        f_test=1, verbose=False):
    """
    Do Pock-Chambolle iterations on train data and stop when MSE on test
    data stops decreasing.
    """
    best_mse = np.sum(Y[mask_test] ** 2) / mask_test.sum()
    mses = np.zeros(max_iter // f_test)
    best_W = np.zeros_like(Y)
    n_non_decrease = 0
    d = mask_train.shape[0]
    sigma = 0.99
    tau = 0.99

    Theta = np.zeros((d, d))
    Theta_old = Theta.copy()
    W = Theta.copy()

    for k in range(max_iter):
        # proximal of nuclear norm: shrink singular values
        U, s, V = np.linalg.svd(W - tau * (2 * Theta - Theta_old),
                                full_matrices=False)
        s = shrink(s, tau)
        W[:] = U @ (s[:, None] * V)
        Theta_old[:] = Theta
        Theta[mask_train] += sigma * (W - Y)[mask_train]

        if k % f_test == 0:
            mse = np.sum((W[mask_test] - Y[mask_test]) ** 2) / mask_test.sum()
            mses[k // f_test] = mse
            if verbose:
                print("Iter %d, mse: %.3f" % (k, mse))
            if mse < best_mse:
                best_mse = mse
                best_W[:] = W
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

    return best_W, mses[:k // f_test + 1]


class LowRankIterReg(LinearModel):
    def __init__(self, train_ratio=0.8, f_test=1, max_iter=1000, memory=100,
                 verbose=False):
        # __super__(Lasso, self).__init__()
        self.train_ratio = train_ratio
        self.f_test = f_test
        self.verbose = verbose
        self.max_iter = max_iter
        self.memory = memory

    def fit(self, mask, Y, train_idx=None, test_idx=None, seed=0):
        if train_idx is None or test_idx is None:
            x1, x2 = np.where(mask)
            n_mask = len(x1)
            np.random.seed(seed)
            perm = np.random.permutation(n_mask)
            train = perm[:int(0.8 * n_mask)]
            test = perm[int(0.8 * n_mask):]
            mask_train = np.zeros_like(mask)
            mask_train[x1[train], x2[train]] = True
            mask_test = np.zeros_like(mask)
            mask_test[x1[test], x2[test]] = True
        else:
            pass

        w, mses = _dual_primal_lowrank_callback(
            mask_train, mask_test, Y, self.max_iter, self.memory,
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
