"""
===============================================
Plot Chambolle Pock path vs Lasso Tikhonov path
===============================================

Comparison of Tikhonov regularization path (fast LassoCV with the celer
package) and iterative regularization (ours).
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from celer import LassoCV
from libsvmdata import fetch_libsvm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

from iterreg.ell1 import dual_primal
from iterreg.ell1.estimators import BasisPursuitIterReg


dataset = 'rcv1_train'
X, y = fetch_libsvm(dataset)

n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

clf = LassoCV(fit_intercept=False, eps=3e-3,
              n_alphas=100, n_jobs=4, cv=kf, verbose=0)

clf.fit(X, y)

max_iter = 500
f_store = 5
n_points = max_iter // f_store
mse_dp = np.zeros((n_points, n_splits))

res = Parallel(n_jobs=-1)(delayed(dual_primal)(
    X[train_idx], y[train_idx], max_iter, f_store, verbose=False)
    for train_idx, _ in kf.split(X))
all_w = np.array([result[-1] for result in res])

for split, (train_idx, test_idx) in enumerate(kf.split(X)):
    mse_dp[:, split] = np.mean(
        (X[test_idx] @ all_w[split].T - y[test_idx, None]) ** 2, axis=0)


best_alpha = clf.alpha_

fig, axarr = plt.subplots(
    2, 1, sharey=True, figsize=(7, 6), constrained_layout=True)
ax = axarr[0]
ax.semilogx(clf.alphas_ / clf.alphas_[0], clf.mse_path_, ':')
ax.semilogx(clf.alphas_ / clf.alphas_[0], clf.mse_path_.mean(axis=-1), 'k',
            linewidth=2)
ax.axvline(clf.alpha_ / clf.alphas_[0], linestyle='--', color='k',
           label=r'best $\lambda$')

ax.set_title("Tikhonov regularization")
ax.set_xticks([1e-2, 1e-1, 1e0])

ax.set_xlabel(r'$\lambda / \lambda_{\mathrm{\max}}$')
ax.set_ylabel('Mean squared error')
ax.legend()

ax = axarr[-1]
ax.set_title("Iterative regularization")
ax.plot(f_store * np.arange(n_points), mse_dp, ':')
ax.plot(f_store * np.arange(n_points), mse_dp.mean(axis=-1), 'k')
best_iter = f_store * np.argmin(np.mean(mse_dp, axis=-1))
ax.axvline(best_iter, linestyle='--', color='k', label='best iteration')
ax.set_xlabel("Chambolle-Pock iteration")
ax.set_ylabel('Mean squared error')
ax.legend()

plt.show(block=False)


# now do the timings a posteriori:
# time if we stopped at best_iter
bp = BasisPursuitIterReg(max_iter=best_iter, memory=best_iter + 1)
t0 = time.perf_counter()
bp.fit(X, y)
time_cp = time.perf_counter() - t0
print("Duration for CP: %.3fs" % time_cp)

# default grid:
alpha_max = np.max(np.abs(X.T @ y)) / len(y)
alphas = np.geomspace(1, 1e-3, 100) * alpha_max
alphas_stop = alphas[alphas >= best_alpha]
# time if we stopped at best_alpha:

t0 = time.perf_counter()
clf = LassoCV(fit_intercept=False, alphas=alphas_stop,
              verbose=0, cv=4, n_jobs=4).fit(X, y)
time_cv = time.perf_counter() - t0


print("CP early stop needs %d iter" % best_iter)
print("LassoCV optimal lambda: %.2e lambda_max" %
      (best_alpha / clf.alphas_[0]))
print("CP time: %.3f s" % time_cp)
print("CV time: %.3f s" % time_cv)
