import jax.numpy as jnp
import numpy as np
from jax import jax, jit
from functools import partial
from timeit import timeit
from copy import deepcopy
from src.line_search import ArmijoLineSearch, WolfeLineSearch

"""
Implementation of Newton CG method
"""

class NewtonConjugateGradientSolver():
    def __init__(self, term_tol = 10**-6, max_iterations = 1000, eta_cg_tol = 0.01, line_searcher = "ArmijoLineSearch"):
        self.term_tol = term_tol # termination tolerance
        self.max_iterations = max_iterations # max iterations
        self.eta_cg_tol = eta_cg_tol # CG termination tolerance
        if line_searcher == "ArmijoLineSearch": # choose line search method
            self.line_searcher = ArmijoLineSearch()
        elif line_searcher == "WolfeLineSearch":
            self.line_searcher = WolfeLineSearch()         

    def set_solver_name(self, name):
        self.name = name # set name

    def check_converge(self, dfdx_k, dfdx_0, iter): # check gradient norm for termination of Newton iterations
        return jnp.linalg.norm(dfdx_k) <= (self.term_tol) * max(1, jnp.linalg.norm(dfdx_0))

    def cg_solve(self, objective_fn, xk): # CG solve
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        
        zj = 0 # initial newton direction
        dfdx = objective_fn.dfdx(xk) # residual
        gradient_eval += 1 # gradient evaluation counnter
        rj = dfdx # residual
        dj = -rj # initial cg direction (steepest descent)
        iter = 0 # iteration
        pk = -dfdx # computed newton direction
        hessian = objective_fn.hessian(xk) # hessian
        while iter <= jnp.size(rj) - 1:
            if jnp.transpose(dj) @ hessian @ dj <= 0: # negative curvature of the hessian along the current search direction, terminate CG solve
                if iter == 0:
                    pk = -objective_fn.dfdx(xk)
                    gradient_eval += 1
                else:
                    pk = zj
                print("Negative curvature encountered, exit CG solve!") # print info
                break
            else:
                alpha_j = (jnp.transpose(rj) @ rj) / (jnp.transpose(dj) @ hessian @ dj) # compute alpha
                zj += alpha_j * dj # update newton direction
                previous_rj = deepcopy(rj)
                rj += alpha_j * hessian @ dj # update residual
                if jnp.linalg.norm(rj) <= self.eta_cg_tol * jnp.linalg.norm(dfdx): # terminate if residual smaller than threshold
                    pk = zj
                    print("CG solve terminates nicely!")
                    break
                beta_j = (jnp.transpose(rj) @ rj) / (jnp.transpose(previous_rj) @ previous_rj) # update beta
                dj = -rj + beta_j * dj # update cg direction
            iter += 1
        if iter >= jnp.size(rj):
            print("CG solve exceeds max iter!")
        return pk, function_eval, gradient_eval

    def newton_cg_solve(self, objective_fn, x0):
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        xk = x0 # initial point
        dfdx_0 = objective_fn.dfdx(xk) # gradient at current point
        gradient_eval += 1 # gradient evaluation counter
        dfdx_k = dfdx_0
        iter = 0 # iteration counter
        fx_traj = [objective_fn.f(xk)] # fx path
        dfdx_traj = [jnp.linalg.norm(dfdx_k)] # gradient path

        while (not self.check_converge(dfdx_k, dfdx_0, iter)) and (iter <= self.max_iterations - 1): # check convergence or max iteration
            
            # search direction
            pk, func_eval_num, gradient_eval_num = self.cg_solve(objective_fn, xk)
            dfdx_k = objective_fn.dfdx(xk)
            function_eval += func_eval_num # function evaluation counter
            gradient_eval += gradient_eval_num  # gradient evaluation counter

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
            print("Newton CG exceeds max iter!")
        print("Function evaluation: ", function_eval, "Gradient evaluation: ", gradient_eval)
        
        return xk, objective_fn.f(xk), fx_traj, dfdx_traj