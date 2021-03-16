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
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from celer import LassoCV
from celer.datasets import make_correlated_data
from celer.plot_utils import configure_plt

from iterreg import BasisPursuitIterReg
from iterreg.ell1 import primal_dual, dual_primal


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
        axarr[0, i].set_title(f"$\sigma=1 /{step} ||A||$", fontsize=20)

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
A_, b_, x_true = make_correlated_data(
    n_samples=1000, n_features=2000, density=0.01, corr=0., snr=np.inf,
    random_state=0)

A, A_test, b, b_test = train_test_split(A_, b_, test_size=0.25, random_state=0)

clf = BasisPursuitIterReg(verbose=0).fit(A, b)

plt.figure(figsize=(6, 3), constrained_layout=True)
plt.semilogy(clf.mses)
plt.axvline(np.argmin(clf.mses), c='k', linestyle='--', label='best iteration')
plt.axhline(mean_squared_error(A_test @ x_true, b_test),
            c='k', label='oracle MSE')
plt.xlabel("Chambolle-Pock iteration")
plt.ylabel('Left out MSE')
plt.legend()
plt.show(block=False)


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
