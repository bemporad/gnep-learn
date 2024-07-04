<img src="http://cse.lab.imtlucca.it/~bemporad/gnep-learn/images/gnep-learn-logo.png" alt="gnep-learn" width=40%/>

A Python package for solving Generalized Nash Equilibrium Problems by active learning of best-response models.
 
# Contents

* [Package description](#description)

* [Installation](#install)

* [Basic usage](#basic-usage)

* [Contributors](#contributors)

* [Acknowledgments](#acknowledgments)

* [Citing jax-sysid](#bibliography)

* [License](#license)


<a name="description"></a>
## Package description 

**gnep-learn** is a Python package for solving Generalized Nash Equilibrium Problems with $n$ agents by actively learning linear surrogate models of the agents' best responses. The algorithm is based on a centralized entity that probes the agents’ reactions and recursively updates the local linear parametric estimates of the action-reaction mappings, possibly converging to a stationary action profile. 

The package implements the approach described in the following paper:

<a name="cite-FB24"></a>
> [1] F. Fabiani, A. Bemporad, "[An active learning method for solving competitive multi-agent decision-making and control problems](
http://arxiv.org/abs/2212.12561)," submitted for publication. Available on arXiv at <a href="http://arxiv.org/abs/2212.12561">
http://arxiv.org/abs/2212.12561</a>, 2024. [[bib entry](#ref1)]


<a name="install"></a>
## Installation

~~~python
pip install gnep-learn
~~~

<a name="basic-usage"></a>
## Basic usage
Consider a set of agents $j=1,\ldots,n$, each one minimizing a private objective function $J_j(x_j,x_{-j})$, possibly under global constraints $Ax\leq b$, $g(x)\leq 0$, where $x\in R^{n_d}$ is the overall decision vector, $x_j$ is the subvector of $x$ containing the variables decided by agent $j$, $x_{-j}$ are the variables optimized by the remaining agents.
Denoting by $f_j(x_{-j})$ the best response of agent $j$ given the remaining decisions $x_{-j}$, the goal is to find
a stationary profile $x^*$ such that $x^*_j=f_j(x^*_{-j})$.

The central entity proposes iteratively a tentative decision vector $x(k)$ to the agents and collects the corresponding best responses $x_j(k)=f_j(x_{-j}(k))$. Such information is used by the central entity to update affine surrogate models $\hat f_j$ of the best responses by linear Kalman filtering (i.e., by recursive least-squares). During the first `N_init` iterations, the vectors $x(k)$ are generated randomly; afterward, based on the learned best response models, the centralized entity solves a constrained least-squares problem to attempt to find a stationary profile $x(k)$.

Let us set up a simple problem with two agents, each one optimizing a single variable within the range $[-10,10]$. We want to run 10 random sampling steps followed by another 10 active learning steps, for a total of $N=20$ iterations:

~~~python
from gnep_learn import GNEP

n = 2  # number of agents
dim = [1, 1]  # each agent optimizes one variable
N_init = 10 # number of initial random sampling iterations
N = 20 # total number of iterations

def J1(x1,x2):
    return x1**2 -x1*x2 - x1 # function minimized by agent #1
def J2(x2,x1):
    return x2**2 +x1*x2 -x2 # function minimized by agent #2

J = [lambda x1, x2: J1(x1,x2)[0],
     lambda x2, x1: J2(x2,x1)[0]]  # collection of agents' objectives

lbx = -10. * np.ones(2) # lower bound on x used during optimization
ubx = 10. * np.ones(2) # upper bound on x used during optimization

gnep = GNEP(J, dim, lbx=lbx, ubx=ubx) # Create GNEP object

xeq, XX, XXeq, _ = gnep.learn(N=N, N_init=N_init) # Learn equilibrium
~~~

The returned vector `xeq` is a stationary profile. The learning function `learn` also returns the sequence `XX` of tentative equilibria suggested by the central entity and `XXeq` the corresponding best responses. 

~~~python
xeq, XX, XXeq, TH = gnep.learn(N=N, N_init=N_init, save_model_params = True)
~~~
also returns the sequence `TH` of parameters of the affine best response models learned, where `TH[k][j][i]` contains the parameter vector $\theta_{ji}(k)$ of the model at step $k$ of the $i$th component of the best response $x_j$ given $x_{-j}$, namely the estimate $[\hat x_j]_i = [x_{-j}'\ 1]\theta_{ji}(k)$ of $[x_j]_i$.

More generally, each agent can minimize an objective function $\tilde J_j(w_j,x_{-j})$, possibly under private constraints
on the local decision vector $w_j$, and returns the shared decision vector $x_j=f_j(x_{-j})=h_j(w_j)$,
where $h_j$ is some private mapping that we assume here to be linear, i.e., $x_j=W_jw_j$.

See the example files in the `examples` folder for how to set up and solve different generalized Nash equilibrium problems.
                
<a name="contributors"></a>
## Contributors

This package was coded by Alberto Bemporad.


This software is distributed without any warranty. Please cite the paper below if you use this software.

<a name="acknowledgments"></a>
## Acknowledgments


<a name="bibliography"></a>
## Citing `gnep-learn`

<a name="ref1"></a>

```
@article{FB24,
    author = {F. Fabiani and A. Bemporad},
    title={An active learning method for solving competitive multi-agent decision-making and control problems},
    note = {submitted for publication. Also available on arXiv
    at \url{http://arxiv.org/abs/2212.12561}},
    year=2024
}
```

<a name="license"></a>
## License

Apache 2.0

(C) 2024 A. Bemporad
