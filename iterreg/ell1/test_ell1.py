import pytest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_less

from celer.datasets import make_correlated_data

from iterreg.ell1.solvers import dual_primal, primal_dual


@pytest.mark.parametrize("solver", [primal_dual, dual_primal])
def test_dual_primal(solver):
    np.random.seed(0)
    X, y, _ = make_correlated_data(20, 30, random_state=0)
    w, theta, _ = solver(X, y, max_iter=100_000)

    # feasability
    np.testing.assert_array_less(norm(X @ w - y) / norm(y), 1e-9)
    # -X.T @ theta should be subgradient of L1 norm at w
    supp = w != 0
    assert_allclose(- X[:, supp].T @ theta, np.sign(w[supp]))
    assert_array_less(np.abs(X[:, ~ supp].T @ theta), 1. - 1e-9)
