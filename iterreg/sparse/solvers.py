import numpy as np

from numba import njit
from numpy.linalg import norm
from scipy import sparse

from iterreg.utils import shrink, power_method


def primal_dual(X, y, step=1, max_iter=1000, f_store=1, alpha_prec=None,
                verbose=False):
    """Chambolle-Pock algorithm with relaxation parameter equal to 1."""
    n, d = X.shape
    if alpha_prec is not None:
        assert 0 <= alpha_prec <= 2
        sigma = 1. / np.sum(np.abs(X) ** alpha_prec, axis=1)
        tau = 1. / np.sum(np.abs(X) ** (2 - alpha_prec), axis=0)
    else:
        L = power_method(X) if sparse.issparse(X) else norm(X, ord=2)
        # stepsizes such that tau * sigma * L ** 2 < 1
        tau = step / L
        sigma = 0.99 / (step * L)

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


def dual_primal(X, y, max_iter=1000, f_store=10, ret_all=True,
                callback=None, memory=10, step=1, rho=0.99, verbose=False,):
    """Chambolle-Pock algorithm applied to the dual: interpolation on the
    primal variable w.

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        Design matrix.
    y : np.array, shape (n_samples,)
        Observation vector.
    max_iter : int, optional (default=1000)
        Maximum number of Chambolle-Pock iterations.
    f_store : int, optional (default=10)
        Primal iterates are stored every `f_store` iterations.
    ret_all : bool, optional (default=True)
        If True, return all stored primal iterates.
    callback : callable or None, optional (default=None)
        Callable called on primal iterate `w` every `f_store` iterations.
    memory : int, optional (default=10)
        If `callback` is not None and its value did not decrease for the last
        `memory` stored iterates, the algorithm is early stopped.
    step : float, optional (default=1)
        Balances primal and dual stepsizes. If `step=1`, both are equal.
    rho : float, optional (default=0.99)
        The product of the step sizes is smaller than `rho / norm(X, ord=2)**2`
    verbose : bool, optional (default=False)
        Verbosity of the algorithm.

    Returns
    -------
    w : np.array, shape (n_features,)
        Last or best primal iterate.
    theta : np.array, shape (n_samples,)
        Last or best dual iterate.
    crits : np.array, shape (max_iter // f_store,)
        Value of callback along iterations.
    all_w : np.array, shape (max_iter // f_store, n_features)
        Primal iterates every `f_store` iterations. Returned only if
        `ret_all` is True.
    """
    n, d = X.shape
    L = power_method(X) if sparse.issparse(X) else norm(X, ord=2)

    crits = np.zeros(max_iter // f_store)

    if callback is not None:
        best_crit = np.inf
        best_w = np.zeros(d)
        n_non_decrease = 0

    # stepsizes such that tau * sigma * L ** 2 = rho < 1
    tau = np.sqrt(rho) * step / L
    sigma = np.sqrt(rho) / (step * L)
    if ret_all:
        all_w = np.zeros([max_iter // f_store, d])
    w = np.zeros(d)
    theta = np.zeros(n)
    theta_old = np.zeros(n)

    for k in range(max_iter):
        w = shrink(w - tau * X.T @ (2 * theta - theta_old), tau)
        theta_old[:] = theta
        theta += sigma * (X @ w - y)
        if k % f_store == 0:
            if ret_all:
                all_w[k // f_store] = w
            if verbose:
                print("Iter %d" % k)
            if callback is not None:
                crits[k // f_store] = callback(w)
                if crits[k // f_store] < best_crit:
                    n_non_decrease = 0
                    best_crit = crits[k // f_store]
                    best_w = w.copy()
                    best_theta = theta.copy()
                else:
                    n_non_decrease += 1
                if n_non_decrease >= memory:
                    if verbose:
                        print("No improvement for %d iterations"
                              "(best: %d), exit" %
                              (memory * f_store, k - memory * f_store))
                        w, theta = best_w, best_theta
                        crits = crits[:k // f_store + 1]
                        break
    if ret_all:
        return w, theta, crits, all_w
    else:
        return w, theta, crits


@njit
def cd_primal_dual(X, y, max_iter=100, f_store=1, verbose=False):
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
