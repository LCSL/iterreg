import numpy as np

from numpy.linalg import norm
from iterreg.utils import shrink


def dual_primal_low_rank(
        mask, Y, max_iter=1000, f_store=10, sigma=None, limit=None,
        stop_crit=1e-10, verbose=False):
    """Lowest nuclear norm matrix equal to Y on mask.
    mask and Y are np.arrays, shape (d, d).
    """
    d = mask.shape[0]

    if sigma is None:
        sigma = 0.99
        tau = 1.
    else:
        tau = 0.99 / sigma
    Theta = np.zeros((d, d))
    Theta_old = Theta.copy()
    distances = np.zeros(max_iter // f_store)
    W = Theta.copy()

    for k in range(max_iter):
        U, s, V = np.linalg.svd(W - tau * (2 * Theta - Theta_old),
                                full_matrices=False)
        s = shrink(s, tau)
        W[:] = U @ (s[:, None] * V)
        Theta_old = Theta.copy()
        Theta[mask] += sigma * (W - Y)[mask]
        if k % f_store == 0:
            if limit is not None:
                distances[k // f_store] = norm(W - limit)
            feasability = norm((W - Y)[mask])
            if verbose:
                print(f"Iter {k}, feasability: {feasability:.1e}")
            if feasability < stop_crit:
                print(
                    f"Feasability {feasability:.1e} < {stop_crit:.1e}, exit.")
                break

    return W, Theta, distances
