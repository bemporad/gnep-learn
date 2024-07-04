# -*- coding: utf-8 -*-
"""
Solving the generalized Nash equilibrium problem described in [1, Fig. 7], originally proposed in [2, Example A.1].

[1] F. Fabiani and A. Bemporad, “An active learning method for solving competitive multi-agent decision-making and control problems,” 2024, http://arxiv.org/abs/2212.12561. 

[2] F. Facchinei and C. Kanzow, “Penalty methods for the solution of generalized Nash equilibrium problems (with complete test problems),” Institute of Mathematics, University of Würzburg, Tech. Rep., 2009.

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
N_init = 5
N = 15
alpha = 1.e-3  # L2-regularization used in KF
beta = 0.  # learning-rate used by KF
rho = 1.e-6  # regularization term rho*||x||_2^2 when finding an equilibrium
xeq0 = None  # no equilibrium is known a priori


def cost(x, i):
    # Cost function minimized by agent #i, i=0,...,n-1
    return -x[i]/jnp.sum(x)*(1.-jnp.sum(x))


def costj(xj, xnotj, j):
    # Rewrite as a function of x(j) and x(-j)
    x = jnp.hstack((xnotj[0:j], xj[0], xnotj[j:]))
    return cost(x, j)


J = []  # agents' objectives
for j in range(n):
    J.append(partial(costj, j=j))

lbx = 0.01 * np.ones(nvar)
ubx = 100. * np.ones(nvar)
lbx[0] = 0.3  # change lower bound for x1
ubx[0] = 0.5  # change upper bound for x1
Lmin = 0. * np.ones(nvar)
Lmax = 100. * np.ones(nvar)

# Impose linear constraint sum(x)<=1
A = np.ones((1, nvar))
b = np.array(1.).reshape(1, 1)
# the first constraint (i.e., sum(x)<=1) is imposed for all agents
c_index = [[0]]*n

# Create GNEP object
gnep = GNEP(J, dim, alpha=alpha, beta=beta, rho=rho, lbx=lbx,
            ubx=ubx, Lmin=Lmin, Lmax=Lmax, A=A, b=b, c_index=c_index)

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
