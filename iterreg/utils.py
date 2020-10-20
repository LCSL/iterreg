import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz
from numpy.linalg import norm
from sklearn.utils import check_random_state
from sklearn.datasets import fetch_openml
from numba import njit


def plot_legend_apart(ax, figname, ncol=None):
    """Do all your plots with fig, ax = plt.subplots(),
    don't call plt.legend() at the end but this instead"""
    if ncol is None:
        ncol = len(ax.lines)
    fig = plt.figure(figsize=(30, 4), constrained_layout=True)
    fig.legend(ax.lines, [line.get_label() for line in ax.lines], ncol=ncol,
               loc="upper center")
    fig.tight_layout()
    fig.savefig(figname)
    os.system("pdfcrop %s %s" % (figname, figname))
    return fig


@njit
def shrink(u, tau):
    """Soft-thresholding of vector u at level tau > 0."""
    return np.sign(u) * np.maximum(0., np.abs(u) - tau)


def bregman_div(x, y, subgrad=None):
    """Bregman divergence for L1."""
    if subgrad is None:
        subgrad = np.sign(y)
    return norm(x, ord=1) - norm(y, ord=1) - (subgrad * (x - y)).sum()


def make_sparse_data(n, d, rho=0.5, s=None, snr=None, w_type="randn",
                     normalize=False, seed=24):
    """
    Generate X and y as follows:
    - X has shape (n, d), has Gaussain entries with Toeplitz correlation
        with parameter rho.
    - noisy or exact measurements: y = X @ w_true + Gaussian noise
    """
    assert w_type in ("ones", "randn")

    rng = check_random_state(seed=seed)
    corr = rho ** np.arange(d)
    cov = toeplitz(corr)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    if normalize:
        X /= norm(X, axis=0)
    if s is None:
        s = d // 10
    supp = rng.choice(d, s, replace=False)
    w_true = np.zeros(d)

    if w_type == "randn":
        w_true[supp] = rng.randn(s)
    elif w_type == "ones":
        w_true[supp] = 1.
    else:
        raise ValueError('Unknown w_type: %s' % w_type)

    y = X @ w_true
    if snr is not None:
        noise = rng.randn(n)
        y += noise / norm(noise) * norm(y) / snr
    return np.asfortranarray(X), y, w_true


def fetch_leukemia():
    data = fetch_openml("leukemia")
    y = np.array([- 1 if lab == 'ALL' else 1 for lab in data.target])
    X = data.data
    return np.asfortranarray(X), y.astype(float)


def power_method(X, max_iter=100, rtol=1e-6):
    np.random.seed(1)
    u = np.random.randn(X.shape[0])
    v = np.random.randn(X.shape[1])
    spec_norm = 0
    for _ in range(max_iter):
        spec_norm_old = spec_norm
        u[:] = X @ v
        u /= norm(u)
        v[:] = X.T @ u
        spec_norm = norm(v)
        v /= spec_norm
        delta = np.abs(spec_norm - spec_norm_old) / spec_norm
        if delta < rtol:
            break
    else:
        warnings.warn('Did not converge, %s > %s' % (delta, rtol))
    return spec_norm
