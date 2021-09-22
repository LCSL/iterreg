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


d = 100
np.random.seed(0)
mask = np.zeros([d, d], dtype=bool)
idx = np.random.choice(d ** 2, d ** 2 // 5, replace=False)
mask.flat[idx] = True
# rank = d // 10
rank = 5
Y_true = np.random.randn(d, rank) @ np.random.randn(rank, d)
Y_true /= (norm(Y_true, ord="fro") / 20)
Y = Y_true.copy()

Y[~mask] = 0

W_star, Theta, _ = dual_primal_low_rank(
    mask, Y, max_iter=3_000, f_store=10, verbose=1)

print(f"Feasability of W_star : {norm((Y - W_star)[mask]):.2e}")

n_deltas = 10
deltas = np.linspace(1, 15, num=n_deltas)

noise = np.random.randn(d, d)
distances = dict()
f_store = 1

deltas = deltas[3:-1]

for delta in deltas:
    print(delta)
    Y_delta = Y_true + delta * noise / norm(noise)
    sigma = 1 / norm(Y_delta, ord=2)

    x, theta, dist = dual_primal_low_rank(
        mask, Y_delta, max_iter=200, sigma=sigma, verbose=False, f_store=f_store,
        limit=W_star)

    distances[delta] = dist

plt.close('all')
fig1, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(3.8, 2.2))
n_points = 100
for delta in deltas:
    x_plt = f_store * np.arange(len(distances[delta]))
    y_plt = distances[delta] / norm(W_star)
    ax.semilogy(x_plt[:n_points], y_plt[:n_points],
                label=r"$\delta={:.1f}$".format(delta))

paper = True
if paper:
    plt.ylabel(r'$||X_k - {X}^\star|| / ||{X}^\star||$')
    plt.xlabel("iterative regularization iteration $k$")
else:
    plt.ylabel(r'$||W_k - {W}^\star|| / ||{W}^\star||$')
    plt.legend(loc='upper right', ncol=3, fontsize=16)

plt.show(block=False)


if paper:
    fig1.savefig("low_rank_d%d.pdf" % d)
