import numpy as np
from numpy.linalg import norm

from iterreg.ell1.solvers import dual_primal


def test_dual_primal():
    np.random.seed(0)
    X, y = np.random.randn(20, 30), np.random.randn(20)
    w, theta, _ = dual_primal(X, y, max_iter=100000)

    # feasability
    np.testing.assert_array_less(norm(X @ w - y), 1e-13)
    # TODO assert that we have a subgradient in X.T @ theta
