"""
=========================================
Compare CV and CP for Low rank completion
=========================================

Compare explicit and implicit for low rank matrix completion.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from celer.plot_utils import configure_plt

from iterreg.low_rank.solvers import dual_primal_low_rank
from sklearn.model_selection import train_test_split
# configure_plt()


d = 50
np.random.seed(0)
observed = np.zeros([d, d], dtype=bool)
idx = np.random.choice(d ** 2, int(0.6 * d ** 2), replace=False)
observed.flat[idx] = True
# rank = d // 10
rank = 5
Y_true = np.random.randn(d, rank) @ np.random.randn(rank, d)
Y_true /= norm(Y_true, ord="fro")
noise = np.random.randn(*Y_true.shape)
noise /= norm(noise)
Y_true += 0.3 * noise

Y = Y_true.copy()
Y[~observed] = 0

idx_train, idx_test = train_test_split(idx, test_size=0.25)

observed_train = np.zeros_like(observed)
observed_train.flat[idx_train] = True
Y_train = Y.copy()
Y_train[~observed_train] = 0
observed_test = np.zeros_like(observed)
observed_test.flat[idx_test] = True
Y_test = Y.copy()
Y_test[~observed_test] = 0


def pgd_nucnorm(Y, alpha, max_iter=10_000, tol=1e-6, X_init=None):
    norm_Y = norm(Y)
    if X_init is None:
        X = Y.copy()
    else:
        X = X_init.copy()
    for it in range(max_iter):
        X_old = X.copy()
        X -= observed * (X - Y)
        U, s, VT = np.linalg.svd(X)
        s = np.sign(s) * np.maximum(np.abs(s) - alpha, 0)
        X = U @ (s[:, np.newaxis] * VT)
        delta = norm(X - X_old)
        if it % 100 == 0:
            print(f"Iter {it}, delta={delta:.2e}")
        if delta <= tol * norm_Y:
            print(f"Early exit at iter {it}.")
            break
    return X


x, _, _, all_x = dual_primal_low_rank(
    observed, Y_train, max_iter=100, f_store=1, retall=True)

losses_cp = [norm((Y_test - x)[observed_test]) / norm(Y_test) for x in all_x]


alpha_max = np.linalg.norm(Y, ord=2)
X = Y_train.copy()
rhos = np.geomspace(1, 1e-3, 20)
all_X_cv = []
for idx, rho in enumerate(rhos):
    X = pgd_nucnorm(Y_train, alpha=rho * alpha_max, X_init=X)
    all_X_cv.append(X.copy())
all_X_cv = np.array(all_X_cv)

losses_cv = [norm((Y_test - x)[observed_test]) / norm(Y_test) for x in all_X_cv]


plt.close('all')
fig, axarr = plt.subplots(
    1, 2, constrained_layout=True, sharey=True, figsize=(7.4, 2))
ax = axarr[0]
ax.semilogx(rhos, losses_cv)
ax.invert_xaxis()
ax.set_xlabel(r"$\lambda / \lambda_{\mathrm{max}}$")
ax.set_ylabel(r"Loss on left out indices")

ax = axarr[1]
ax.plot(np.arange(len(losses_cp)), losses_cp)
ax.set_xlabel("Chambolle-Pock iteration")

plt.show(block=False)
