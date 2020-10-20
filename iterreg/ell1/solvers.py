import numpy as np

from numba import njit
from numpy.linalg import norm
from scipy import sparse

from iterreg.utils import shrink, power_method


def primal_dual(X, y, max_iter=100, f_store=1, alpha_prec=None,
                verbose=False):
    """Chambolle-Pock algorithm with theta=1."""
    n, d = X.shape
    if alpha_prec is not None:
        assert 0 <= alpha_prec <= 2
        sigma = 1. / np.sum(np.abs(X) ** alpha_prec, axis=1)
        tau = 1. / np.sum(np.abs(X) ** (2 - alpha_prec), axis=0)
    else:
        if sparse.issparse(X):
            tau = 1 / power_method(X, max_iter=1000)
        else:
            tau = 1 / norm(X, ord=2)
        sigma = tau
    all_w = np.zeros([max_iter // f_store, d])
    w = np.zeros(d)
    w_bar = np.zeros(d)
    theta = np.zeros(n)

    for k in range(max_iter):
        theta += sigma * (X @ w_bar - y)
        w_old = w.copy()
        w = shrink(w - tau * (X.T @ theta), tau)
        w_bar[:] = 2 * w - w_old
        if k % f_store == 0:
            all_w[k // f_store] = w
            if verbose:
                print("Iter %d" % k)
    return w, theta, all_w


def dual_primal(X, y, max_iter=1000, f_store=10, verbose=False):
    n, d = X.shape
    if sparse.issparse(X):
        tau = 1 / power_method(X)
    else:
        tau = 1 / norm(X, ord=2)
    sigma = tau
    all_w = np.zeros([max_iter // f_store, d])
    w = np.zeros(d)
    theta = np.zeros(n)
    theta_old = np.zeros(n)

    for k in range(max_iter):
        w = shrink(w - tau * X.T @ (2 * theta - theta_old), tau)
        theta_old[:] = theta
        theta += sigma * (X @ w - y)
        if k % f_store == 0:
            all_w[k // f_store] = w
            if verbose:
                print("Iter %d" % k)

    return w, theta, all_w


@njit
def cd(X, y, max_iter=100, f_store=1, verbose=False):
    n, d = X.shape
    taus = 1. / (2. * (X ** 2).sum(axis=0))
    res = - y  # residuals: Ax - b
    w = np.zeros(d)
    all_w = np.zeros((max_iter // f_store, d))
    theta = np.zeros(n)
    theta_bar = np.zeros(n)
    for t in range(max_iter):
        for j in range(d):
            theta_bar = theta + res / d  # doing stuff inplace would be faster
            old = w[j]
            w[j] = shrink(w[j] - taus[j] * X[:, j] @
                          (2 * theta_bar - theta), taus[j])
            theta *= (1. - 1. / d)
            theta += theta_bar / d
            if w[j] != old:
                res += (w[j] - old) * X[:, j]
        if t % f_store == 0:
            all_w[t // f_store] = w
            if verbose:
                print("Iter ", t)

    return w, theta, all_w


@njit
def ST(x, u):
    if x > u:
        return x - u
    elif x < -u:
        return x + u
    else:
        return 0.


@njit
def cd_lasso(X, y, alpha, max_iter, f_store=1):
    p = X.shape[1]
    lc = np.zeros(p)
    for j in range(p):
        lc[j] = norm(X[:, j]) ** 2
    R = y.copy().astype(np.float64)
    w = np.zeros(p)
    E = np.zeros(max_iter // f_store)
    all_w = np.zeros((max_iter // f_store, p))

    for t in range(max_iter):
        for j in range(p):
            old = w[j]
            w[j] = ST(old + X[:, j].dot(R) / lc[j], alpha / lc[j])
            if w[j] != old:
                R += ((old - w[j])) * X[:, j]
        if t % f_store == 0:
            E[t // f_store] = (R ** 2).sum() / 2. + alpha * np.sum(np.abs(w))
            all_w[t // f_store] = w
            print(t, E[t // f_store])

    return w, all_w, E


@njit
def ista_lasso(X, y, alpha, max_iter, f_store=1):
    p = X.shape[1]
    L = norm(X, ord=2) ** 2
    w = np.zeros(p)
    E = np.zeros(max_iter // f_store)
    R = y.copy().astype(np.float64)
    all_w = np.zeros((max_iter // f_store, p))

    for t in range(max_iter):
        R[:] = y - X @ w
        tmp = w + 1. / L * X.T @ R
        w[:] = shrink(tmp, alpha / L)
        if t % f_store == 0:
            E[t // f_store] = (R ** 2).sum() / 2. + alpha * np.sum(np.abs(w))
            all_w[t // f_store] = w
            print(t, E[t // f_store])

    return w, all_w, E


@njit
def fista_lasso(X, y, alpha, max_iter, f_store=1):
    p = X.shape[1]
    L = norm(X, ord=2) ** 2
    w = np.zeros(p)
    z = np.zeros(p)
    t_new = 1
    E = np.zeros(max_iter // f_store)
    R = y.copy().astype(np.float64)
    all_w = np.zeros((max_iter // f_store, p))

    for t in range(max_iter):
        w_old = w.copy()
        w[:] = shrink(z - X.T @ (X @ z - y) / L, alpha / L)
        t_old = t_new
        t_new = (1. + np.sqrt(1 + 4 * t_old ** 2)) / 2.
        z[:] = w + (t_old - 1.) / t_new * (w - w_old)

        if t % f_store == 0:
            E[t // f_store] = ((X @ w - y) ** 2).sum() / \
                2. + alpha * np.sum(np.abs(w))
            all_w[t // f_store] = w
            print(t, E[t // f_store])
    return w, all_w, E
