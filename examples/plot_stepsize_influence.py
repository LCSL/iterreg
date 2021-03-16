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


configure_plt(fontsize=16)

###############################################################################
# An util plot function:


def plot_varying_sigma(A, b, x_true, steps, max_iter=100):
    _, axarr = plt.subplots(3, 3, sharey='row', sharex=True, figsize=(9, 6),
                            constrained_layout=True)

    for i, step in enumerate(steps):
        _, _, _, all_x = dual_primal(
            A, b, step=step, ret_all=True, max_iter=max_iter, f_store=1)
        scores = [f1_score(x != 0, x_true != 0) for x in all_x]
        supp_size = np.sum(all_x != 0, axis=1)
        axarr[0, i].set_title(r"$\sigma=1 /%d ||A||$" % step, fontsize=20)

        axarr[0, i].plot(scores)
        axarr[0, i].set_ylim(0, 1)
        axarr[0, i].axvline(np.argmax(scores), c='k',
                            linestyle='--', label='max value')
        if i == 0:
            axarr[0, i].set_ylabel('F1 score for support')
        axarr[0, i].legend()

        axarr[1, i].plot(supp_size)
        if i == 0:
            axarr[1, i].set_ylabel(r"$||x_k||_0$")

        axarr[2, i].plot(norm(all_x - x_true, axis=1))
        if i == 0:
            axarr[2, i].set_ylabel(r'$\Vert x_k - x^*\Vert$')
        axarr[2, i].set_xlabel("CP iteration")
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


###############################################################################
# On last example, compare with Lasso

def f1(clf, X, y):
    return f1_score(clf.coef_ != 0, x_true != 0)


clf = Lasso(fit_intercept=False)
alphas = norm(A.T @ b, ord=np.inf) / len(b) * np.geomspace(1, 1e-3)
grid_search = GridSearchCV(
    clf, {'alpha': alphas}, scoring=f1, refit=False, cv=3).fit(A, b)

plt.figure(figsize=(5, 3), constrained_layout=True)
plt.semilogx(alphas, grid_search.cv_results_["mean_test_score"])
plt.xlabel(r'$\lambda$')
plt.ylabel("F1 score for support")
plt.title("Lasso regularization path")
plt.show(block=False)
