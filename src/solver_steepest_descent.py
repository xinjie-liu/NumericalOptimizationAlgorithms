import jax.numpy as jnp
import numpy as np
from jax import jax, jit
from functools import partial
from timeit import timeit
from copy import deepcopy
from src.line_search import ArmijoLineSearch, WolfeLineSearch

"""
Implementation of steepest descent method
"""

class SteepestDescent():
    def __init__(self, term_tol = 10**-6, max_iterations = 1000, line_searcher = "ArmijoLineSearch"):
        self.term_tol = term_tol # termination tolerance
        self.max_iterations = max_iterations  # max iterations
        if line_searcher == "ArmijoLineSearch": # choose line search method
            self.line_searcher = ArmijoLineSearch()
        elif line_searcher == "WolfeLineSearch":
            self.line_searcher = WolfeLineSearch()

    def set_solver_name(self, name):
        self.name = name # set name
    
    # @partial(jit, static_argnums=0)
    def check_converge(self, dfdx_k, dfdx_0, iter): # check gradient norm and max iterations for termination of Newton iterations
        return jnp.linalg.norm(dfdx_k) <= (self.term_tol) * max(1, jnp.linalg.norm(dfdx_0)) or iter >= self.max_iterations

    def steepest_descent_solve(self, objective_fn, x0):
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        xk = x0 # initial point
        dfdx_0 = objective_fn.dfdx(xk) # gradient at current point
        dfdx_k = dfdx_0
        gradient_eval += 1 # gradient evaluation counter
        iter = 0  # iteration counter
        converge_flag = False
        fx_traj = [objective_fn.f(xk)]  # fx path
        dfdx_traj = [jnp.linalg.norm(dfdx_k)] # gradient path

        # args for computing initial step length
        xk_1 = x0
        dfdx_k_1 = dfdx_0
        pk_1 = -dfdx_k_1

        while not converge_flag:
            # search direction
            dfdx_k = objective_fn.dfdx(xk)
            pk = -dfdx_k
            gradient_eval += 1 # gradient evaluation counter
            converge_flag = self.check_converge(dfdx_k, dfdx_0, iter)
            # step length (initial step length is iteration dependent)
            alpha_t = 2*(objective_fn.f(xk) - objective_fn.f(xk_1))/(dfdx_k_1@pk_1 + 1e-8) if iter > 0 else 1
            function_eval += 1
            if alpha_t < 0: # use initial step size 1 if the above gives ill-posed step size
                alpha_t = 1
            alpha_k, func_eval_num, gradient_eval_num = self.line_searcher.search_stepsize(objective_fn, xk, pk, alpha_t) # line search for step size
            function_eval += func_eval_num # function evaluation counter
            gradient_eval += gradient_eval_num  # gradient evaluation counter
            
            # store quantities for next iter
            xk_1 = deepcopy(xk)
            dfdx_k_1 = deepcopy(dfdx_k)
            pk_1 = -dfdx_k_1

            # take a step
            xk += alpha_k * pk

            if iter % 100 == 0 or converge_flag:
                print("Iter: ", iter, "f(x): ", objective_fn.f(xk), "||dfdx||: ", jnp.linalg.norm(dfdx_k), "alpha: ", alpha_k)
            
            fx_traj.append(objective_fn.f(xk))
            dfdx_traj.append(jnp.linalg.norm(dfdx_k))
            iter += 1

        print("Function evaluation: ", function_eval, "Gradient evaluation: ", gradient_eval)
        
        return xk, objective_fn.f(xk), fx_traj, dfdx_traj