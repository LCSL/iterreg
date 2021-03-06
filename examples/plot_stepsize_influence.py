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
from sklearn.model_selection import train_test_split
from celer.datasets import make_correlated_data
from celer.homotopy import celer_path

from iterreg.sparse import dual_primal


n_samples = 500
n_features = 1_000

###############################################################################
# The function to compute CP, Lasso path and plot metrics:


def plot_varying_sigma(corr, density, snr, steps, max_iter=100, rho=0.99):
    np.random.seed(0)
    # true coefficient vector has entries equal to 0 or 1
    supp = np.random.choice(n_features, size=int(density * n_features),
                            replace=False)
    w_true = np.zeros(n_features)
    w_true[supp] = 1
    X_, y_, w_true = make_correlated_data(
        n_samples=int(n_samples * 4 / 3.), n_features=n_features,
        w_true=w_true,
        corr=corr, snr=snr, random_state=0)

    X, X_test, y, y_test = train_test_split(X_, y_, test_size=0.25)

    print('Starting computation for this setting')
    fig, axarr = plt.subplots(4, 2, sharey='row', sharex='col',
                              figsize=(7, 5), constrained_layout=True)

    fig.suptitle(r"Correlation=%.1f, $||w^*||_0$= %s, snr=%s" %
                 (corr, (w_true != 0).sum(), snr))

    for i, step in enumerate(steps):
        _, _, _, all_w = dual_primal(
            X, y, step=step, rho=rho, ret_all=True, max_iter=max_iter,
            f_store=1)
        scores = [f1_score(w != 0, w_true != 0) for w in all_w]
        supp_size = np.sum(all_w != 0, axis=1)
        mses = [mean_squared_error(y_test, X_test @ w) for w in all_w]

        axarr[0, 0].plot(scores, label=r"$\sigma=1 /%d ||X||$" % step)
        axarr[1, 0].semilogy(supp_size)
        axarr[2, 0].plot(norm(all_w - w_true, axis=1))
        axarr[3, 0].plot(mses)

    axarr[0, 0].set_ylim(0, 1)
    axarr[0, 0].set_ylabel('F1 score for support')
    axarr[1, 0].set_ylabel(r"$||w_k||_0$")
    axarr[2, 0].set_ylabel(r'$\Vert w_k - w^*\Vert$')
    axarr[2, 0].set_xlabel("CP iteration")
    axarr[3, 0].set_ylabel("pred MSE left out")
    axarr[0, 0].legend(loc='lower right', fontsize=10)
    axarr[0, 0].set_title('Iterative regularization')

    # last column: Lasso results
    alphas = norm(X.T @ y, ord=np.inf) / len(y) * np.geomspace(1, 1e-3)

    coefs = celer_path(X, y, 'lasso', alphas=alphas)[1].T
    axarr[0, 1].semilogx(
        alphas, [f1_score(coef != 0, w_true != 0) for coef in coefs])
    axarr[1, 1].semilogx(
        alphas, [np.sum(coef != 0) for coef in coefs])
    axarr[2, 1].semilogx(
        alphas, [norm(coef - w_true) for coef in coefs])
    axarr[3, 1].semilogx(
        alphas, [mean_squared_error(y_test, X_test @ coef) for coef in coefs])

    axarr[3, 1].set_xlabel(r'$\lambda$')
    axarr[0, 1].set_title("Lasso path")

    for i in range(3):
        axarr[i, 1].set_xlim(*axarr[i, 1].get_xlim()[::-1])

    plt.show(block=False)


###############################################################################
# Noiseless case where RIP holds (L1 sol = L0 sol)
density = 0.01
corr = 0.
snr = np.inf

plot_varying_sigma(corr, density, snr, [2, 10, 100], max_iter=100)
###############################################################################
# A different setting, with more correlation in A but still noiseless

corr = 0.5
snr = np.inf
density = 0.1
plot_varying_sigma(corr, density, snr, [2, 10, 100], max_iter=100)


###############################################################################
# Now if in addition w_true is less sparse, L1 solution is no longer L0 sol

corr = 0.5
snr = np.inf
density = 0.5

plot_varying_sigma(corr, density, snr, [2, 10, 100], max_iter=100)


###############################################################################
# Finally, when there is noise in the data:
corr = 0.2
density = 0.1
snr = 10

plot_varying_sigma(corr, density, snr, [2, 10, 100], max_iter=100)


###############################################################################
# And when the product of the stepsizes is not taken maximal:
# tau * sigma = rho

rho = 0.1

plot_varying_sigma(corr, density, snr, [2, 10, 100], rho=rho, max_iter=100)
