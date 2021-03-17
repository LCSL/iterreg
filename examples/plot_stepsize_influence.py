"""
=======================================================
Influence of primal-dual stepsizes for support recovery
=======================================================

This example shows that using a smaller dual stepsize slows down convergence,
but is beneficial for support recovery.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from celer import Lasso
from celer.datasets import make_correlated_data
from celer.plot_utils import configure_plt

from iterreg.ell1 import dual_primal


configure_plt()

###############################################################################
# An util plot function:


def f1(clf, X, y):
    return f1_score(clf.coef_ != 0, x_true != 0)


def support_size(clf, X, y):
    return np.sum(clf.coef_ != 0)


def plot_varying_sigma(A, b, x_true, steps, max_iter=100):
    _, axarr = plt.subplots(3, 2, sharey='row', sharex='col',
                            figsize=(7, 5), constrained_layout=True)

    def distance_x_true(clf, X_, y):
        return norm(clf.coef_ - x_true)

    for i, step in enumerate(steps):
        _, _, _, all_x = dual_primal(
            A, b, step=step, ret_all=True, max_iter=max_iter, f_store=1)
        scores = [f1_score(x != 0, x_true != 0) for x in all_x]
        supp_size = np.sum(all_x != 0, axis=1)

        axarr[0, 0].plot(scores, label=r"$\sigma=1 /%d ||A||$" % step)
        axarr[1, 0].semilogy(supp_size)
        axarr[2, 0].plot(norm(all_x - x_true, axis=1))

    axarr[0, 0].set_ylim(0, 1)
    axarr[0, 0].set_ylabel('F1 score for support')
    axarr[1, 0].set_ylabel(r"$||x_k||_0$")
    axarr[2, 0].set_ylabel(r'$\Vert x_k - x^*\Vert$')
    axarr[2, 0].set_xlabel("CP iteration")
    axarr[0, 0].legend()

    # last column: Lasso results
    clf = Lasso(fit_intercept=False)
    alphas = norm(A.T @ b, ord=np.inf) / len(b) * np.geomspace(1, 1e-3)
    grid_search = GridSearchCV(
        clf, {'alpha': alphas},
        scoring={'f1': f1,
                 'supp': support_size,
                 'dist_x_true': distance_x_true},
        refit=False, cv=3).fit(A, b)

    axarr[0, 1].semilogx(
        alphas, grid_search.cv_results_["mean_test_f1"])
    axarr[1, 1].semilogx(
        alphas, grid_search.cv_results_["mean_test_supp"])
    axarr[2, 1].semilogx(
        alphas, grid_search.cv_results_["mean_test_dist_x_true"])
    for i in range(3):
        axarr[i, 1].set_xlim(*axarr[i, 1].get_xlim()[::-1])
    axarr[2, 1].set_xlabel(r'$\lambda$')
    axarr[0, 1].set_title("Lasso path")
    plt.show(block=False)


###############################################################################
# Noiseless case where RIP holds (L1 sol = L0 sol)
A, b, x_true = make_correlated_data(
    n_samples=500, n_features=1000, density=0.01, corr=0., snr=np.inf,
    random_state=0)

plot_varying_sigma(A, b, x_true, [2, 10, 100], max_iter=100)
###############################################################################
# A different setting, with more correlation in A but still noiseless

corr = 0.5
snr = np.inf
A, b, x_true = make_correlated_data(
    n_samples=1000, n_features=2000, density=0.1, corr=corr, snr=snr,
    random_state=0)
plot_varying_sigma(A, b, x_true, [2, 10, 100], max_iter=100)


###############################################################################
# Now if in addition x_true is less sparse, L1 solution is no longer L0 sol

corr = 0.5
snr = np.inf
density = 0.5

A, b, x_true = make_correlated_data(
    n_samples=1000, n_features=2000, density=density, corr=corr, snr=snr,
    random_state=0)
plot_varying_sigma(A, b, x_true, [2, 10, 100], max_iter=100)


###############################################################################
# Finally, when there is noise in the data:
corr = 0.2
density = 0.1
snr = 10

A, b, x_true = make_correlated_data(
    n_samples=1000, n_features=2000, density=density, corr=corr, snr=snr,
    random_state=0)
plot_varying_sigma(A, b, x_true, [2, 10, 100], max_iter=100)
