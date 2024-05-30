import jax.numpy as jnp
import numpy as np
from jax import jax, jit
from functools import partial
from timeit import timeit
from copy import deepcopy
from src.line_search import ArmijoLineSearch, WolfeLineSearch

"""
Implementation of Newton's method and modified Newton's method
"""

class NewtonSolver():
    def __init__(self, term_tol = 10**-6, max_iterations = 1000, beta = 10**-4, modify_hessian_flag = False, line_searcher = "ArmijoLineSearch"):
        self.term_tol = term_tol # termination tolerance
        self.max_iterations = max_iterations # max iterations
        self.beta = beta # beta for Hessian modification
        self.modify_hessian_flag = modify_hessian_flag # whether or not modify Hessian
        if line_searcher == "ArmijoLineSearch": # choose line search method
            self.line_searcher = ArmijoLineSearch()
        elif line_searcher == "WolfeLineSearch":
            self.line_searcher = WolfeLineSearch()
    
    def set_solver_name(self, name):
        self.name = name # set name

    def check_converge(self, dfdx_k, dfdx_0, iter):  # check gradient norm for termination of Newton iterations
        return jnp.linalg.norm(dfdx_k) <= (self.term_tol) * max(1, jnp.linalg.norm(dfdx_0)) 

    def modify_hessian(self, hessian): # modify Hessian by Cholesky with added multiple of identity
        beta = self.beta
        min_diag = jnp.min(jnp.diag(hessian)) # minimum element of the diagonal of the Hessian
        # initial delta
        delta_t = -min_diag + beta if min_diag <= 0 else 0

        break_loop = False
        while not break_loop:
            try: # add identity and try Cholesky factorization. If successful, terminate; otherwise, add more identity
                break_loop = True
                modified_hessian = hessian + delta_t * jnp.identity(jnp.shape(hessian)[0])
                L = np.linalg.cholesky(modified_hessian)
            except:
                break_loop = False
            if not break_loop:
                delta_t = max(2 * delta_t, beta)
        print("delta_t: ", delta_t)
        # TODO: downstream linear system of equations solve can use upstream L for speed optimization
        return modified_hessian # return the modified Hessian

    def newton_solve(self, objective_fn, x0):
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        modify_hessian = self.modify_hessian_flag
        xk = x0 # initial point
        dfdx_0 = objective_fn.dfdx(xk)  # gradient at current point
        dfdx_k = dfdx_0
        gradient_eval += 1 # gradient evaluation counter
        iter = 0 # iteration counter
        fx_traj = [objective_fn.f(xk)] # fx path
        dfdx_traj = [jnp.linalg.norm(dfdx_k)] # gradient path

        while (not self.check_converge(dfdx_k, dfdx_0, iter)) and (iter <= self.max_iterations - 1): # check convergence or max iteration
            # search direction
            dfdx_k = objective_fn.dfdx(xk) # gradient
            hessian = objective_fn.hessian(xk) # hessian
            gradient_eval += 1  # gradient evaluation counter
            if modify_hessian:
                hessian = self.modify_hessian(hessian) # modify hessian if required
            # pk = -jnp.linalg.inv(hessian)@dfdx_k # explicit Hessian inversion
            pk = jnp.linalg.solve(hessian, -dfdx_k) # compute Newton's direction by solving system of equations

            # step length
            alpha_k, func_eval_num, gradient_eval_num = self.line_searcher.search_stepsize(objective_fn, xk, pk, 1.0) 
            function_eval += func_eval_num # function evaluation counter
            gradient_eval += gradient_eval_num  # gradient evaluation counter
            
            # take a step
            xk += alpha_k * pk

            print("Iter: ", iter, "f(x): ", objective_fn.f(xk), "||dfdx||: ", jnp.linalg.norm(dfdx_k), "alpha: ", alpha_k)
            fx_traj.append(objective_fn.f(xk))
            dfdx_traj.append(jnp.linalg.norm(dfdx_k))
            iter += 1
        
        if iter >= self.max_iterations: # print info
            print("Newton solve exceeds max iter!")
        print("Function evaluation: ", function_eval, "Gradient evaluation: ", gradient_eval)

        return xk, objective_fn.f(xk), fx_traj, dfdx_traj