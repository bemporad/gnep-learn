# -*- coding: utf-8 -*-
"""
gnep_learn: an active learning method to solve generalized Nash equilibrium problems.

(C) 2024 A. Bemporad
"""

import numpy as np
from scipy.optimize import linprog, lsq_linear
from cvxopt import matrix, solvers
from functools import partial
import jax.numpy as jnp
import jax
import jaxopt
import copy
import time
import tqdm
from numba import njit
from numba.typed import List
from joblib import Parallel, delayed
from scipy.stats.qmc import LatinHypercube


@jax.jit  # this is a bit faster than numpy code
def KF_update_(th, P, xij, xnotj, beta):
    """
    Kalman filter update step for estimating the parameters of the affine best response models
    via recursive least-squares.

    Args:
        th (ndarray): Model parameter vector.
        P (ndarray): Covariance matrix of the parameter vector.
        xij (float): measured component #i of the best response of agent #j.
        xnotj (ndarray): current values expressed by the other agents #h, h=1,...,j-1,j+1,...n
        beta (float): Coefficient defining the variance of process noise, determining the learning-rate of the estimator.

    Returns:
        tuple: Updated model parameter vector and covariance matrix.

    (C) 2024 A. Bemporad
    """
    R = 1.  # output noise variance, corresponding to the loss |xij_k-xij_hat_k|^2
    C = jnp.hstack((xnotj, 1.)).reshape(1, -1)  # row vector
    PC = P @ C.T
    # note: this is ok because the measurement xij is a scalar
    M = PC / (R + C @ PC)
    e = xij-jnp.sum(th*C)
    th += (M*e).reshape(-1)  # update model parameter vector
    # update covariance matrix for all parameter vectors defining [x(j)]_i
    P = P - M @ PC.T  # P(k|k). This is the same as P = P - M@(C@P)
    P = (P + P.T) / 2. + beta * jnp.eye(P.shape[0])  # P(k+1|k)
    return th, P


@njit
def build_LS(th, nvar, n, dim, notj):
    """
    Builds the matrices of the least-squares problem used by the central entity to propose a new equilibrium.

    Parameters:
    th (list): List of arrays representing the coefficients of the best response models.
    nvar (int): Number of variables.
    n (int): Number of agents.
    dim (list): List of dimensions of the best response of each agent.
    notj (list): List of indices determining the entries within the overall response vector x
    of the decisions taken by the other agents.

    Returns:
    tuple: A tuple containing the coefficient matrix C and the vector d defining the least-squares objective.

    (C) 2024 A. Bemporad
    """

    C = np.eye(nvar)
    d = np.zeros(nvar)
    h = 0
    for j in range(n):
        for i in range(dim[j]):
            # min ||x[j]-th[j][0:n-1]@x(-j)-th[j][n-1]||_2^2
            C[h, notj[j]] = -th[j][i, :-1]
            d[h] = th[j][i, -1]
            h += 1
    return C, d


class BR():
    """Collection of affine best-response models x(j) = th[j]*[x(-j);1], j=1,...,n, 
    with n = number of agents, and th[j] is a matrix with dim[j] rows and 
    sum(dim[j])-dim[j] columns.

    (C) 2024 A. Bemporad
    """

    def __init__(self, dim):
        """Best response models.

        Parameters:
        ----------
        dim : list of int
            number of components of x(j) for j = 1,...,n, where n=len(dim) is the number of agents.
        """
        self.dim = dim  # number of variables per agent
        self.n = len(dim)  # number of agents
        # total number of shared decision variables, i.e., dimension of vector x
        self.nvar = np.sum(dim)
        self.notj = list()  # components of x(-j) within overall vector x
        self.nother = list()  # dimension of x(-j)
        self.params = list()  # parameters defining the affine best-response models
        # covariance matrices of the parameters defining the affine best-response models
        self.P = list()
        # index of first component of x(j) within x
        self.i1 = np.empty(self.n, dtype=int)
        # index of last component of x(j), i.e., x(j)=x[i1:i2]
        self.i2 = np.empty(self.n, dtype=int)
        # list of the indices of the components of x(j) within x, i.e., [i1[j],...,i2[j]-1]
        self.isj = list()

        i1 = 0
        i2 = 0
        for j in range(self.n):
            i2 += dim[j]
            notj = np.concatenate((np.arange(0, i1), np.arange(i2, self.nvar)))
            self.i1[j] = i1
            self.i2[j] = i2
            self.isj.append(np.arange(i1, i2, dtype=int))
            i1 = i2
            nother = notj.size
            self.nother.append(nother)
            self.notj.append(notj)
            self.params.append(np.zeros((dim[j], nother + 1)))
        return

    def eval(self, j, x):
        """
        Evaluate the estimated best response of agent j at x.

        Parameters:
        - j (int): The index of the agent.
        - x (numpy.ndarray): The overall vector of agents' decisions.

        Returns:
        - numpy.ndarray: The estimated best response of agent j at x.
        """
        return self.params[j] @ np.hstack((x[self.notj[j]], 1.))

    def KF_update(self, j, i, xij, xnotj, beta):
        """Kalman filter update of model(j,i) based on new best response.

        Args:
            j (int): Index of the agent.
            i (int): Index of the component #i of agent #j to be updated.
            xij (float): Value of [x(j)]_i.
            xnotj (ndarray): Value of x(-j).
            beta (float): Coefficient defining the variance of process noise, defining the learning-rate of the estimator. As the variance of output measurement noise is fixed equal to 1, the larger is 1/beta, the slower is the observer.

        Returns:
            None

        (C) 2024 A. Bemporad
        """
        self.params[j][i], self.P[j][i] = KF_update_(
            self.params[j][i], self.P[j][i], xij, xnotj, beta)

    def KF_initialize(self, alpha):
        """
        Initializes the Kalman filter by setting the initial covariance matrices.

        Parameters:
        - alpha: Scalar determining the initial covariance matrices P(0)= 1./alpha * I.
        This corresponds to imposing the L2-regularization term alpha*||th||_2^2 on
        the vector th of model coefficients.

        Returns:
        None
        """
        for j in range(self.n):
            self.P.append([jnp.eye((self.nother[j] + 1)) /
                          alpha for i in range(self.dim[j])])
        return


class GNEP():
    """Generalized Nash equilibrium problem.

    (C) 2024 A. Bemporad
    """

    def __init__(self, J, dim, rho=1.e-6, alpha=1.e-5, beta=0.,
                 lbx=None, ubx=None, Lmin=None, Lmax=None,
                 lbw=None, ubw=None, w2x=None,
                 A=None, b=None, c_index=None,
                 g=None, g_index=None,
                 gamma=1.e4, verbose=1, lbfgs_maxiter=1000, lbfgs_iprint=-1, lbfgs_memory=20, lbfgs_tol=1.e-8):
        """
        Initializes a Generalized Nash equilibrium problem.

        Args:
            J (list): List of cost functions, one per agent.
            dim (list): List of dimensions of the decisions shared by each agent.
            rho (float, optional): L2-regularization parameter used by central entity when proposing a new decision vector (penalty: rho*||x||_2^2).
            alpha (float, optional): Covariance matrices of best response model parameters initialization, P(0) = 1/alpha*I.
            beta (float, optional): Coefficient defining the variance of process noise used in the Kalman filter, defining the learning-rate of the estimator (the smaller is beta, the faster is the observer).
            lbx (ndarray, optional): Lower bounds on the vector of decision variables used during optimization.
            ubx (ndarray, optional): Upper bounds on the vector of decision variables used during optimization.
            Lmin (ndarray, optional): Lower bounds on local decision variables used during initialization (passive learning).
            Lmax (ndarray, optional): Upper bounds on local decision variables used during initialization (passive learning).
            lbw (list, optional): Lower bounds on local decision variables.
            ubw (list, optional): Upper bounds on local decision variables.
            w2x (list, optional): Mapping from local decision variables to shared decision variables.
            A (ndarray, optional): Linear constraint matrix.
            b (ndarray, optional): Linear constraint vector.
            c_index (list, optional): Indices of linear constraints imposed by each agent.
            g (list, optional): List of nonlinear constraint functions.
            g_index (list, optional): Indices of nonlinear constraints imposed by each agent.
            gamma (float, optional): Penalty parameter for violation of nonlinear constraints.
            verbose (int, optional): Verbosity level.
            lbfgs_maxiter (int, optional): Maximum number of iterations for L-BFGS solver when used to compute best responses.
            lbfgs_iprint (int, optional): Printing frequency for L-BFGS solver.
            lbfgs_memory (int, optional): Limited-memory number used by L-BFGS solver.
            lbfgs_tol (float, optional): Tolerance used by L-BFGS solver.

        Returns:
            None

        (C) 2024 A. Bemporad
        """

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        n = len(dim)  # number of agents
        nvar = np.sum(dim)  # number of variables

        if lbx is None:
            lbx = -100. * np.ones(nvar)
        if ubx is None:
            ubx = 100. * np.ones(nvar)
        if Lmin is None:
            Lmin = lbx-0.5*np.abs(lbx)
        if Lmax is None:
            Lmax = ubx+0.5*np.abs(ubx)

        # Handle linear constraints
        if A is None:
            A = np.zeros((0, nvar))
            b = np.zeros((0, 1))
        if c_index is None:
            # by default, all linear constraints are imposed in each agent's problem
            c_index = [np.arange(b.size)] * n

        # Handle nonlinear constraints
        if g is None:
            isnonlinearconstr = False
            self.J_eq = None
        else:
            isnonlinearconstr = True
            if g_index is None:
                # by default, all nonlinear constraints are imposed in each agent's problem
                g_index = [np.arange(len(g))] * n

            # function to be minimized by the centralized entity to find an equilibrium
            @jax.jit
            def J_eq(x, C, d):
                cost = (jnp.sum((C@x-d)**2)+rho*jnp.sum(x**2))/2.
                for k in range(len(g)):
                    viol = g[k](x)
                    cost += gamma*jnp.maximum(viol, 0.0)**2
                if islinearconstr:
                    viol = A@x.reshape(-1, 1)-self.b
                    cost += gamma*jnp.sum(jnp.maximum(viol, 0.0)**2)
                return cost
            self.J_eq = J_eq

        # Create collection of best response models
        self.BR = BR(dim)
        self.BR.KF_initialize(alpha)  # initialize covariance matrices

        # Handle box constraints on local decision variables
        if w2x is None:
            w2x = [None]*n
        if lbw is None:
            lbw = [None]*n
        if ubw is None:
            ubw = [None]*n
        for j in range(n):
            if w2x[j] is None:  # w[j]=x[j]
                lbw[j] = lbx[self.BR.isj[j]]
                ubw[j] = ubx[self.BR.isj[j]]
        isweqx = [w2x[j] is None for j in range(n)]

        islinearconstr = (b.size > 0)
        if islinearconstr:
            self.Aj = list()
            self.bj = list()
            self.Sj = list()
            for j in range(n):
                self.Aj.append(A[c_index[j]][:, self.BR.isj[j]].reshape(
                    np.array(c_index[j]).size, self.BR.dim[j]))
                self.bj.append(b[c_index[j]])
                self.Sj.append(A[c_index[j]][:, self.BR.notj[j]].reshape(
                    np.array(c_index[j]).size, self.BR.nother[j]))
        else:
            self.Aj = None
            self.bj = None
            self.Sj = None

        def JJ(wj, xnotj, j):
            cost = J[j](wj, xnotj)
            if isweqx[j]:
                xj = wj
            else:
                xj = w2x[j]@wj
            if islinearconstr:
                viol = self.Aj[j]@xj.reshape(-1, 1) + \
                    self.Sj[j]@xnotj.reshape(-1, 1)-self.bj[j]
                cost += gamma*jnp.sum(jnp.maximum(viol, 0.0)**2)
            if isnonlinearconstr:
                x = jnp.empty(nvar)
                x = x.at[self.BR.isj[j]].set(xj)
                x = x.at[self.BR.notj[j]].set(xnotj)
                for k in range(g_index[j].size):
                    viol = g[g_index[j][k]](x)
                    cost += gamma*jnp.maximum(viol, 0.0)**2
            return cost
        self.J = JJ

        # Use L-BFGS solver to compute best responses
        self.lbfgs_options = {'iprint': min(lbfgs_iprint, 90), 'maxls': 20, 'gtol': lbfgs_tol, 'eps': 1.e-8,
                              'ftol': lbfgs_tol, 'maxcor': lbfgs_memory}
        self.lbfgs_maxiter = lbfgs_maxiter

        # Store parameters
        self.lbx = lbx
        self.ubx = ubx
        self.w2x = w2x
        self.isweqx = isweqx
        self.lbw = lbw
        self.ubw = ubw
        self.A = A
        self.b = b
        self.c_index = c_index
        self.g = g
        self.g_index = g_index
        self.rho = rho
        self.sqrtrho = np.sqrt(rho)
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.n = n
        self.nvar = nvar
        self.beta = beta
        self.alpha = alpha
        self.islinearconstr = islinearconstr
        self.isnonlinearconstr = isnonlinearconstr
        self.gamma = gamma  # penalty on constraint violation
        self.verbose = verbose
        return

    def best_response(self, j, x):
        """
        Compute the best response x(j) given x(-j).

        Parameters:
            j (int): The index of the agent for which the best response is computed.
            x (ndarray): The current strategy profile of all agents.

        Returns:
            ndarray: The best response of agent #j.

        Notes:
            This method uses the L-BFGS-B algorithm to find the best response strategy.

        (C) 2024 A. Bemporad
        """
        notj = self.BR.notj[j]
        isj = self.BR.isj[j]
        xnotj = x[notj].reshape(-1)

        Jj = partial(self.J, xnotj=xnotj, j=j)
        if self.isweqx[j]:
            w = jnp.array(x[isj])
        else:
            w = jnp.array((self.lbw[j]+self.ubw[j])/2.)
        # this gets modified by jaxopt!
        opts = copy.deepcopy(self.lbfgs_options)
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)
        solver = jaxopt.ScipyBoundedMinimize(fun=Jj, tol=opts["ftol"], method="L-BFGS-B",
                                             options=opts, maxiter=self.lbfgs_maxiter)
        w, _ = solver.run(w, bounds=(self.lbw[j], self.ubw[j]))
        w = np.array(w)

        if self.isweqx[j]:
            xjopt = w
        else:
            xjopt = self.w2x[j]@w
        return xjopt

    def equilibrium(self, x0, dim=None, notj=None):
        """
        Candidate equilibrium solution computed by the centralized entity by constrained least squares.`

        Parameters:
        - x0: Initial guess for the solution.
        - dim: List of dimensions of the best response of each agent.
        - notj: List of indices determining the entries within the overall response vector x of the other agents' decisions.

        Returns:
        - x: The equilibrium solution.
        - fopt: The optimal objective function value (sum of least-squares residuals).

        The equilibrium solution is obtained by minimizing the objective function
        sum(||x_j-fhat_j(x(-j))||_2^2) + 0.5*rho*||x||_2^2, where fhat_j(x(-j)) is
        the best response estimate x_j=th[i]@[x(-j);1].

        If there are no linear or nonlinear constraints, the problem is solved using
        the lsq_linear function from scipy.optimize. If there are linear constraints,
        the problem is solved using the qp function from CVXOPT. Otherwise, the problem
        is solved using the L-BFGS-B algorithm.

        Note: The specific solver used depends on the constraints specified during
        initialization of the class.

        """
        rho = self.rho
        nvar = self.nvar

        if dim is None:
            # create a typed.List copy of self.BR.dim and self.BR.notj
            dim = List()
            [dim.append(x) for x in self.BR.dim]
        if notj is None:
            notj = List()
            [notj.append(x) for x in self.BR.notj]

        C, d = build_LS(np.array(self.BR.params), self.nvar,
                        self.n, dim, notj)

        if not self.islinearconstr and not self.isnonlinearconstr:
            res = lsq_linear(np.vstack((C, self.sqrtrho*np.eye(nvar))),
                             np.hstack((d, np.zeros(nvar))), bounds=(self.lbx, self.ubx))
            x = res.x
            fopt = res.cost
        elif self.islinearconstr and not self.isnonlinearconstr:
            # use CVXOPT
            Qmat = matrix(C.T @ C + rho*np.eye(self.nvar))
            cmat = matrix(-C.T @ d)
            Amat = matrix(
                np.block([[self.A], [np.eye(nvar)], [-np.eye(nvar)]]))
            rhs = self.b.copy()
            bmat = matrix(
                np.block([[rhs], [self.ubx.reshape(-1, 1)], [-self.lbx.reshape(-1, 1)]]))
            sol = solvers.qp(Qmat, cmat, Amat, bmat, options={
                             "show_progress": False})
            x = np.array(sol["x"]).reshape(-1)
            fopt = sol["primal objective"]
        else:
            # use L-BFGS-B to solve the problem
            loss = partial(self.J_eq, C=C, d=d)
            # this gets modified by jaxopt!
            opts = copy.deepcopy(self.lbfgs_options)
            solver = jaxopt.ScipyBoundedMinimize(
                fun=loss, tol=opts["ftol"], method="L-BFGS-B", options=opts, maxiter=self.lbfgs_maxiter)
            x, state = solver.run(jnp.array(x0), bounds=(self.lbx, self.ubx))
            x = np.array(x)
            fopt = state.fun_val

        return x, fopt

    def learn(self, N=100, N_init=30, use_parallel=True, use_LHS=False, save_model_params=False):
        """
        Learning-based method to determine an equilibrium of a Generalized Nash Equilibrium Problem.

        Parameters:
        - N (int): The total number of iterations to perform, i.e., of queries to the agents.
        - N_init (int): The number of initial iterations performed during the initial random phase (passive learning).
        - use_parallel (bool): If True, the best responses of the agents are computed in parallel.
        - use_LHS (bool): If True, the initial decision vectors are generated using Latin Hypercube Sampling, otherwise they are drawn randomly from the uniform distribution. The generated vectors are projected on the feasible set before querying the agents.
        - save_model_params (bool): If True, the history of the parameters of the best response models is saved and returned.

        Returns:
        - xk (ndarray): The final decision vector proposed by the centralized optimizer.
        - XX (ndarray): The history of decision vectors proposed by the centralized entity at each iteration.
        - XXeq (ndarray): The history of best responses of the agents, where XXeq[k] is the vector of best responses at the decision vector XX[k].


        (C) 2024 A. Bemporad
        """

        DL = self.Lmax - self.Lmin
        nvar = self.nvar
        XX = np.zeros((N, nvar))
        XXeq = np.zeros((N, nvar))
        n = self.n
        TH = list()

        if self.verbose:
            iters_tqdm = tqdm.tqdm(total=N, desc='Iterations', ncols=40,
                                   bar_format='{percentage:3.0f}%|{bar}|', leave=True, position=0)
            iters_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')

        t0 = time.time()

        # create a typed.List copy of self.BR.dim and self.BR.notj
        notj = List()
        [notj.append(x) for x in self.BR.notj]
        dim = List()
        [dim.append(x) for x in self.BR.dim]

        if use_LHS:
            # Latin Hypercube Sampling
            lhd = LatinHypercube(d=self.nvar)
            Xref_init = self.Lmin + DL * lhd.random(n=N_init)

        for k in range(N):
            if k < N_init:
                # get initial feasible move by minimizing ||x-xref||_inf -> eps>=x(i)-xref(i), eps>=-x(i)+xref(i)
                if not use_LHS:
                    xref = self.Lmin + DL * np.random.rand(nvar)
                else:
                    xref = Xref_init[k]

                c = np.zeros(nvar + 1)
                c[0] = 1.
                lb = np.hstack((0, self.lbx))
                ub = np.hstack((np.inf, self.ubx))
                A_ub = np.block([[-np.ones((nvar, 1)), np.eye(nvar)],
                                [-np.ones((nvar, 1)), -np.eye(nvar)],
                                [np.zeros((self.A.shape[0], 1)), self.A]])
                res = linprog(c, A_ub=A_ub, b_ub=np.block([[xref.reshape(
                    nvar, 1)], [-xref.reshape(nvar, 1)], [self.b]]), bounds=np.vstack((lb, ub)).T, method='highs')
                if res.x is None:
                    raise (Exception(res.message))
                else:
                    x = res.x[1:]
                txt = 'random init:    '

            else:
                x, fopt = self.equilibrium(x, dim, notj)
                txt = 'active learning:'

            if self.verbose:
                txt2 = "x = "
                x_print = x[0:min(self.nvar, 4)]
                txt2 += np.array2string(x_print, precision=4,
                                        separator=',', max_line_width=80)[:-1]
                if self.nvar > 4:
                    txt2 += ", ..."
                else:
                    txt2 += "]"
                iters_log.set_description_str(f"{txt} {txt2}")
                iters_tqdm.update(1)

            # Compute best-responses for each agent #j and update parameters via KF
            xk = x.copy()  # decision vector proposed by centralized optimizer
            XXeq[k] = xk
            if use_parallel:
                def br_and_update(j, xk, paramsj, Pj):
                    if not jax.config.jax_enable_x64:
                        # Enable 64-bit computations
                        jax.config.update("jax_enable_x64", True)
                    xj = self.best_response(j, xk)
                    xnotj = xk[self.BR.notj[j]].reshape(-1)
                    for i in range(self.BR.dim[j]):
                        paramsj[i], Pj[i] = KF_update_(
                            paramsj[i], Pj[i], xj[i], xnotj, self.beta)
                        # self.BR.KF_update(j, i, xj[i], xnotj, self.beta)
                    return [xj, paramsj, Pj]

                res = Parallel(n_jobs=min(10, self.n))(delayed(br_and_update)(
                    j, xk, self.BR.params[j], self.BR.P[j]) for j in range(self.n))
                for j in range(self.n):
                    x[np.arange(self.BR.i1[j], self.BR.i2[j])] = res[j][0]
                    self.BR.params[j] = res[j][1]
                    self.BR.P[j] = res[j][2]
            else:
                # Non-parallel version
                for j in range(self.n):
                    xj = self.best_response(j, xk)
                    x[np.arange(self.BR.i1[j], self.BR.i2[j])] = xj
                    xnotj = xk[self.BR.notj[j]].reshape(-1)
                    for i in range(self.BR.dim[j]):
                        self.BR.KF_update(j, i, xj[i], xnotj, self.beta)

            XX[k] = x
            if save_model_params:
                TH.append(copy.deepcopy(self.BR.params))

        if self.verbose:
            iters_tqdm.close()
            iters_log.close()

        self.time = time.time()-t0
        return xk, XX, XXeq, TH
