from jax import jax, jit
from functools import partial
import jax.numpy as jnp
import numpy as np

"""
This file contains implementation of different problem classes, which are all inherented from the parent class `ObjectiveFunction`. We use JAX to auto-differentiate the functions to get gradient and Hessian. 
"""

#======================================== Base & Utils ==========================================#

def generate_symmetric_matrix(smallest_eig, largest_eig, n, rng):
    """
    Randomly generate a PSD matrix with given condition number eig_max / eig_min with dimension of n
    """
    eigenvalues = np.linspace(smallest_eig, largest_eig, n)
    Q = rng.random((n, n))
    Q, _ = np.linalg.qr(Q)
    A = Q.dot(np.diag(eigenvalues)).dot(Q.T)
    return A

class ObjectiveFunction():
    """
    This class implements fx, gradient and hessian of fx
    It is intentionally set up without internal parameters to keep static for jit
    This will precompile gradient and Hessian computation so that they run fast online
    """
    @partial(jit, static_argnums=0)
    def dfdx(self, x): # function gradient 
        return jax.grad(self.f)(x)
    
    @partial(jit, static_argnums=0)
    def hessian(self, x): # function Hessian
        return jax.jacfwd(jax.grad(self.f))(x)
    
def construct_problem(name, x0):
    """
    Initialize a problem object with the user specified problem name and starting point
    """
    if name == "Rosenbrock_2":
        problem = Problem1()
    elif name == "Rosenbrock_100":
        problem = Problem2()
    else: # raise an exception if name is out of the scope
        raise Exception("Please pass a proper problem name! Options: ['Rosenbrock_2', 'Rosenbrock_100']")

    problem.set_initial_point(x0) # set initial point
    
    return problem

#======================================== Project Problems ==========================================#

# All problem classes have common fields `name` and `x0` indicating the probelm name string and the initial guess
# The field `x0` needs to be set by invoking the `self.set_initial_point` method

class Problem1(ObjectiveFunction):
    def __init__(self):
        self.dim_n = 2
        self.name = "Rosenbrock_2"

    def set_initial_point(self, x0):
        # set up initial point
        if not jnp.shape(x0)[0] == self.dim_n:
            raise Exception("The initial guess x0 should have the correct dimension!")
        else:
            self.x0 = x0
    
    @partial(jit, static_argnums=0)
    def f(self, x):
        fx = 0.0
        for ii in range(self.dim_n - 1):
            fx += 100 * (x[ii+1] - x[ii]**2)**2 + (1 - x[ii])**2
        return fx

class Problem2(ObjectiveFunction):
    def __init__(self):
        self.dim_n = 100
        self.name = "Rosenbrock_100"

    def set_initial_point(self, x0):
        # set up initial point
        if not jnp.shape(x0)[0] == self.dim_n:
            raise Exception("The initial guess x0 should have the correct dimension!")
        else:
            self.x0 = x0
    
    @partial(jit, static_argnums=0)
    def f(self, x):
        fx = 0.0
        for ii in range(self.dim_n - 1):
            fx += 100 * (x[ii+1] - x[ii]**2)**2 + (1 - x[ii])**2
        return fx
