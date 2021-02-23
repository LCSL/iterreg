"""
=================================================
Plot stopping time for low rank matrix completion
=================================================

Plot the stopping time for low rank matrix completion.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from celer.plot_utils import configure_plt

from iterreg.low_rank.solvers import dual_primal_low_rank

configure_plt()


d = 20
np.random.seed(0)
mask = np.zeros([d, d], dtype=bool)
idx = np.random.choice(d ** 2, d ** 2 // 5, replace=False)
mask.flat[idx] = True

Y_true = np.random.randn(d, 5) @ np.random.randn(5, d)
Y_true /= (norm(Y_true, ord="fro") / 20)
Y = Y_true.copy()

Y[~mask] = 0

W_star, Theta, _ = dual_primal_low_rank(
    mask, Y, max_iter=20000, f_store=100, verbose=0)


n_deltas = 10
deltas = np.linspace(1, 15, num=n_deltas)

noise = np.random.randn(d, d)
distances = dict()
f_store = 1

for delta in deltas:
    print(delta)
    Y_delta = Y + delta * noise / norm(noise)
    w, theta, dist = dual_primal_low_rank(
        mask, Y_delta, max_iter=1000, verbose=False, f_store=f_store,
        limit=W_star)

    distances[delta] = dist

fig1, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 4))
n_points = 500
for delta in deltas[3:-1]:
    x_plt = f_store * np.arange(len(distances[delta]))
    y_plt = distances[delta] / norm(W_star)
    ax.semilogy(x_plt[:n_points], y_plt[:n_points],
                label=r"$\delta={:.1f}$".format(delta))
plt.ylabel(r'$||w_k^\delta - w^\star|| / ||w^\star||$')
plt.xlabel("Iteration $k$")
plt.legend(loc='upper right', ncol=3, fontsize=16)
plt.show(block=False)
