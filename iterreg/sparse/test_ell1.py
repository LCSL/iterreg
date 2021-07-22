import pytest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_less
from celer.datasets import make_correlated_data
from celer import Lasso

from iterreg import SparseIterReg
from iterreg.sparse.solvers import dual_primal, cd, ista, fista, reweighted
from iterreg.utils import deriv_MCP


@pytest.mark.parametrize("solver", [dual_primal])
def test_dual_primal(solver):
    np.random.seed(0)
    X, y, _ = make_correlated_data(20, 30, random_state=0)
    w, theta, _ = solver(X, y, max_iter=100_000, ret_all=False)

    # feasability
    np.testing.assert_array_less(norm(X @ w - y) / norm(y), 1e-9)
    # -X.T @ theta should be subgradient of L1 norm at w
    supp = w != 0
    assert_allclose(- X[:, supp].T @ theta, np.sign(w[supp]))
    assert_array_less(np.abs(X[:, ~ supp].T @ theta), 1. - 1e-9)


def test_BP():
    np.random.seed(0)
    X, y, _ = make_correlated_data(200, 300, random_state=0)
    clf = SparseIterReg(verbose=True, f_test=1, memory=30).fit(X, y)
    np.testing.assert_equal(np.argmin(clf.mses),
                            len(clf.mses) - clf.memory - 1)


def test_cd_ista_fista():
    np.random.seed(0)
    X, y, _ = make_correlated_data(20, 40, random_state=0)
    alpha = np.max(np.abs(X.T @ y)) / 5
    w, _, _ = cd(X, y, alpha, max_iter=100)
    clf = Lasso(fit_intercept=False, alpha=alpha/len(y)).fit(X, y)

    np.testing.assert_allclose(w, clf.coef_, atol=5e-4)

    w, _, _ = ista(X, y, alpha, max_iter=1_000)
    np.testing.assert_allclose(w, clf.coef_, atol=5e-4)

    w, _, _ = fista(X, y, alpha, max_iter=1_000)
    np.testing.assert_allclose(w, clf.coef_, atol=5e-4)


def test_cd_warm_start():
    X, y, _ = make_correlated_data(30, 50, random_state=12)
    alpha = np.max(np.abs(X.T @ y)) / 100

    # same to do 20 iter, or 10 iter, and 10 iter again with warm start:
    for algo in [cd, ista]:
        w, _, E = algo(X, y, alpha, max_iter=20, f_store=1)

        w, _, E1 = algo(X, y, alpha, max_iter=10, f_store=1)
        w, _, E2 = algo(X, y, alpha, w_init=w, max_iter=10, f_store=1)
        np.testing.assert_allclose(E, np.hstack([E1, E2]))


def test_rw_cvg():
    X, y, _ = make_correlated_data(20, 40, random_state=0)
    alpha = np.max(np.abs(X.T @ y)) / 5
    w, E = reweighted(X, y, alpha, max_iter=1000, n_adapt=5)
    clf = Lasso(fit_intercept=False, alpha=alpha/len(y)).fit(X, y)

    np.testing.assert_allclose(w, clf.coef_, atol=5e-4)

    np.testing.assert_allclose(E[-1] / E[0], E[-2] / E[0], atol=5e-4)

    w, E = reweighted(X, y, alpha, deriv_MCP)

    np.testing.assert_allclose(E[-1] / E[0], E[-2] / E[0], atol=5e-4)
