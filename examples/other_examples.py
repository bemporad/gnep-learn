# -*- coding: utf-8 -*-
"""
Examples of generalized Nash equilibrium problems solved by active learning.

(C) 2024 A. Bemporad
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gnep_learn import GNEP

plotfigs = True

np.random.seed(4)  # for reproducibility

example = 0
# example = 1
# example = 2
# example = 3
# example = 4

rho = 1.e-6  # regularization term rho*||x||_2^2 when finding an equilibrium
alpha = 1.e-5  # L2-regularization used in KF
beta = 0.  # learning-rate used by KF
A = None
b = None
c_index = None
w2x = None
lbw = None
ubw = None
gamma = 1.e3
g = None
g_index = None
xeq0 = None  # known Nash equilibrium

if example == 0:
    n = 2  # number of agents
    dim = [1, 1]  # 2 agents of dimension 1
    nvar = 2
    N_init = 10
    N = 20

    def J1(x1, x2):
        return x1**2 - x1*x2 - x1

    def J2(x2, x1):
        return x2**2 + x1*x2 - x2

    J = [lambda x1, x2: J1(x1, x2)[0],
         lambda x2, x1: J2(x2, x1)[0]]  # agents' objectives

    lbx = -10. * np.ones(2)  # lower bound on x used during optimization
    ubx = 10. * np.ones(2)  # upper bound on x used during optimization
    Lmin = None
    Lmax = None

elif example == 1:
    def Jcoop(x1, x2):
        # Cooperative objective function
        J = .5 * (x1 ** 2 + 0.5 * (x1 * x2) + 2 * x2 ** 2) + x1 + x2
        return J

    J = [lambda x1, x2: Jcoop(x1, x2)[0],
         lambda x2, x1: Jcoop(x1, x2)[0]]  # agents' objectives

    # gnep_learn parameters
    n = 2  # number of agents
    dim = [1, 1]  # 2 agents of dimension 1
    nvar = np.sum(dim)
    N_init = 10  # min(50,int(N/2))
    N = 20
    alpha = 1.e-5  # L2-regularization used in KF
    beta = 0.  # learning-rate used by KF

    lbx = -100. * np.ones(nvar)  # lower bound on x used during optimization
    ubx = 100. * np.ones(nvar)  # upper bound on x used during optimization
    # lower bound on x used during initialization (passive learning)
    Lmin = -10 * np.ones(nvar)
    # upper bound on x used during initialization (passive learning)
    Lmax = 10. * np.ones(nvar)

elif example == 2:
    def cost1(x1, xnotj):
        return jnp.sum((x1+xnotj-10.)**2)+jnp.sum(x1**2)

    def cost2(x2, xnotj):
        return jnp.sum((xnotj+x2-10.)**2)+jnp.sum(x2**2)

    J = [cost1, cost2]  # agents' objectives

    # GNE_LEARN parameters
    n = 2  # number of agents
    dim = [10, 10]  # 2 agents of dimension 10
    nvar = np.sum(dim)
    N_init = 10
    N = 20

    lbx = -100. * np.ones(nvar)
    ubx = 100. * np.ones(nvar)
    Lmin = -10. * np.ones(nvar)
    Lmax = 10. * np.ones(nvar)

    # Linear inequality constraints A*x<=b
    A = np.vstack((np.ones((1, nvar)), -np.ones((1, nvar))))
    b = np.array([100.+1.e-6, -100.+1.e-6]).reshape(2, 1)
    c_index = [[0, 1], [0, 1]]

elif example == 3:
    def cost1(x1, xnotj):
        return jnp.sum(x1**2)+2.*jnp.sum((xnotj-2.)*x1)

    def cost2(x2, xnotj):
        return 9.*jnp.sum(x2**2)-2.*jnp.sum((xnotj-2.)*x2)

    J = [cost1, cost2]  # agents' objectives

    # GNE_LEARN parameters
    n = 2  # number of agents
    dim = [200, 200]  # 2 agents of dimension 200
    nvar = np.sum(dim)
    N_init = 3  # min(50,int(N/2))
    N = 10

    lbx = -1. * np.ones(nvar)
    ubx = 1. * np.ones(nvar)
    Lmin = -1. * np.ones(nvar)
    Lmax = 1. * np.ones(nvar)

elif example == 4:
    # In this example, the agents' optimization variables w do not coincide with the shared variables x
    n = 2  # number of agents
    dim = [10, 10]  # 2 agents of dimension 10
    nvar = np.sum(dim)
    # mapping from w to x for each agent
    w2x = [2.*np.eye(dim[0]), 2.*np.eye(dim[1])]

    def cost1(w1, xnotj):
        return jnp.sum((w2x[0]@w1+xnotj-10.)**2)+jnp.sum((w2x[0]@w1)**2)

    def cost2(w2, xnotj):
        return jnp.sum((xnotj+w2x[1]@w2-10.)**2)+jnp.sum((w2x[1]@w2)**2)

    J = [cost1, cost2]  # agents' objectives

    N_init = 10  # min(50,int(N/2))
    N = 20

    lbx = -100. * np.ones(nvar)
    ubx = 100. * np.ones(nvar)
    Lmin = -10. * np.ones(nvar)
    Lmax = 10. * np.ones(nvar)

    lbw = [lbx[:dim[0]]/2., lbx[dim[0]:]/2.]
    ubw = [ubx[:dim[0]]/2., ubx[dim[0]:]/2.]

    if 1:
        # impose constraints as linear constraints
        A = np.vstack((np.ones((1, nvar)), -np.ones((1, nvar))))
        b = np.array([100.+1.e-6, -100.+1.e-6]).reshape(2, 1)
        c_index = [[0, 1], [0, 1]]  # impose both constraints on both agents
    else:
        # impose constraints as general nonlinear constraints
        g = [lambda x: jnp.sum(x)-100.-1.e-6, lambda x: -jnp.sum(x)+100-1.e-6]
        c_index = [[0, 1], [0, 1]]  # impose both constraints on both agents
        gamma = 1.e4
        rho = 1.e-3

else:
    raise ValueError('Unknown example')

# Create GNEP object
gnep = GNEP(J, dim, alpha=alpha, beta=beta, rho=rho, lbx=lbx, ubx=ubx, Lmin=Lmin, Lmax=Lmax, A=A,
            b=b, c_index=c_index, verbose=2, w2x=w2x, lbw=lbw, ubw=ubw, gamma=gamma, g=g, g_index=g_index)

# Learn equilibrium
xeq, XX, XXeq, TH = gnep.learn(N=N, N_init=N_init, save_model_params=True)

print(f"Elapsed time: {gnep.time: .2f} s")

# Compute the best responses at the estimated equilibrium
xhat = np.empty(nvar)
for j in range(0, n):
    xhat[gnep.BR.isj[j]] = gnep.best_response(j, xeq)  # best response

print(f"Estimated equilibrium x: {xeq}")
print(f"Best response @x:        {xhat}")
if np.linalg.norm(xeq-xhat)/np.linalg.norm(xeq) <= 1.e-3:
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
