# -*- coding: utf-8 -*-
"""
Solving the generalized Nash equilibrium problem described in [1, Fig. 6], originally proposed in [2, Section 5] for n=20 agents.

[1] F. Fabiani and A. Bemporad, “An active learning method for solving competitive multi-agent decision-making and control problems,” 2024, http://arxiv.org/abs/2212.12561. 

[2] F. Salehisadaghiani, W. Shi, and L. Pavel, “An ADMM approach to the problem of distributed Nash equilibrium seeking.” CoRR, 2017.

(C) 2024 A. Bemporad
"""

import numpy as np
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gnep_learn import GNEP

plotfigs = True

np.random.seed(0)  # for reproducibility

# gnep_learn parameters
n = 10  # number of agents
dim = [1]*n  # n agents of dimension 1
nvar = np.sum(dim)
N_init = 10  # one must increase N_init and N if n is increased
N = 25
alpha = 1.e-6  # L2-regularization used in KF
beta = 0.  # learning-rate used by KF
rho = 1.e-6  # regularization term rho*||x||_2^2 when finding an equilibrium
xeq0 = None  # no equilibrium is known a priori


def cost(x, i):
    # Cost function minimized by agent #i, i=0,...,n-1
    ci = n*(1.+i/2.)
    return ci*x[i]-x[i]*(60.*n-jnp.sum(x))


def costj(xj, xnotj, j):
    # Rewrite as a function of x(j) and x(-j)
    x = jnp.hstack((xnotj[0:j], xj[0], xnotj[j:]))
    return cost(x, j)


J = []  # agents' objectives
for j in range(n):
    J.append(partial(costj, j=j))

lbx = 7. * np.ones(nvar)
ubx = 100. * np.ones(nvar)
Lmin = 7. * np.ones(nvar)
Lmax = 100. * np.ones(nvar)

# Create GNEP object
gnep = GNEP(J, dim, alpha=alpha, beta=beta, rho=rho,
            lbx=lbx, ubx=ubx, Lmin=Lmin, Lmax=Lmax)

# Learn equilibrium
xeq, XX, XXeq, TH = gnep.learn(N=N, N_init=N_init)

print(f"Elapsed time: {gnep.time: .2f} s")

# Compute the best responses at the estimated equilibrium
xhat = np.empty(nvar)
for j in range(0, n):
    xhat[gnep.BR.isj[j]] = gnep.best_response(j, xeq)  # best response

print(f"Estimated equilibrium x: {xeq}")
print(f"Best response @x:        {xhat}")
if np.linalg.norm(xeq-xhat)/np.linalg.norm(xeq) <= 1.e-2:
    print("*** Equilibrium found! ***")
if xeq0 is not None:
    print(f"Known equilibrium: {xeq0}")

if plotfigs:
    plt.close("all")
    plt.ion()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(np.arange(1, N+1), XXeq, label='predicted')
    ylim = ax.get_ylim()
    rect = patches.Rectangle(
        (0, ylim[0]), N_init, ylim[1]-ylim[0], facecolor='gray', alpha=0.3)
    ax.add_patch(rect)
    ax.grid()
    ax.set_title('optimization vector')
    plt.show()
