# jax-sysid: A Python package for linear and nonlinear system identification and nonlinear regression using Jax.

import numpy as np
import unittest
from gnep_learn import GNEP

# The inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
#
# Authors: A. Bemporad

class Test_gnep_learn(unittest.TestCase):

    def test_example(self):
        # Solve a given GNEP problem using gnep_learn
        
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

        def Jcoop(x1, x2):
            # Cooperative objective function: min .5*[x1 x2]*[1 .25; .25 2]*[x1; x2] + [1 1]*[x1; x2]
            J = .5 * (x1 ** 2 + 0.5 * (x1 * x2) + 2 * x2 ** 2) + x1 + x2
            return J

        J = [lambda x1, x2: Jcoop(x1, x2)[0],
            lambda x2, x1: Jcoop(x1, x2)[0]]  # agents' objectives

        xeq0 = np.linalg.solve(np.array([[1, .25],[.25, 2]]),-np.array([1,1]))  # known equilibrium
        
        # gnep_learn parameters
        n = 2  # number of agents
        dim = [1, 1]  # 2 agents of dimension 1
        nvar = np.sum(dim)
        N_init = 10  # min(50,int(N/2))
        N = 20
        alpha = 1.e-5  # L2-regularization used in KF
        beta = 0.  # learning-rate used by KF

        lbx = -100. * np.ones(nvar) # lower bound on x used during optimization
        ubx = 100. * np.ones(nvar) # upper bound on x used during optimization
        Lmin = -10 * np.ones(nvar) # lower bound on x used during initialization (passive learning)
        Lmax = 10. * np.ones(nvar) # upper bound on x used during initialization (passive learning)

        # Create GNEP object
        gnep = GNEP(J, dim, alpha=alpha, beta=beta, rho=rho, lbx=lbx, ubx=ubx, Lmin=Lmin, Lmax=Lmax, A=A,
                    b=b, c_index=c_index, verbose=2, w2x=w2x, lbw=lbw, ubw=ubw, gamma=gamma, g=g, g_index=g_index)

        # Learn equilibrium
        xeq, _, _, _ = gnep.learn(N=N, N_init=N_init)

        # Compute the best responses at the estimated equilibrium
        xhat = np.empty(nvar)
        for j in range(0, n):
            xhat[gnep.BR.isj[j]] = gnep.best_response(j, xeq)  # best response

        print(f"Known equilibrium: {xeq0}")
        print(f"Estimated equilibrium x: {xeq}")
        print(f"Best response @x:        {xhat}")
        eq_found = np.linalg.norm(xeq-xhat)/np.linalg.norm(xeq) <= 1.e-3
        if eq_found:
            print("*** Equilibrium found! ***")        

        self.assertEqual(eq_found,True,'Equilibrium not found')
        return


if __name__ == '__main__':
    unittest.main()
