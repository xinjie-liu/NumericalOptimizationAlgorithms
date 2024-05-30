import jax.numpy as jnp
import numpy as np
from jax import jax, jit
from functools import partial
from timeit import timeit
from copy import deepcopy
from src.line_search import ArmijoLineSearch, WolfeLineSearch

"""
Implementation of BFGS Quasi Newton method
"""

class QuasiNewtonBFGSSolver():
    def __init__(self, term_tol = 10**-6, max_iterations = 1000, line_searcher = "ArmijoLineSearch", epsilon = 10**-8):
        self.term_tol = term_tol # termination tolerance
        self.max_iterations = max_iterations # max iterations
        if line_searcher == "ArmijoLineSearch": # choose line search method
            self.line_searcher = ArmijoLineSearch()
        elif line_searcher == "WolfeLineSearch":
            self.line_searcher = WolfeLineSearch()
        self.epsilon = epsilon # criterion constant to determine whether to perform BFGS update

    def set_solver_name(self, name):
        self.name = name # set name

    def check_converge(self, dfdx_k, dfdx_0, iter): # check gradient norm for termination of Newton iterations
        return jnp.linalg.norm(dfdx_k) <= (self.term_tol) * max(1, jnp.linalg.norm(dfdx_0)) 

    def bfgs_update(self, Hk, sk, yk): # perform BFGS update to compute approximation of Hessian inverse
        rho_k = 1 / (jnp.transpose(yk) @ sk)
        identity = jnp.eye(sk.shape[0])
        Hk_prime = (identity - rho_k * jnp.outer(sk, jnp.transpose(yk))) @ Hk @ (identity - rho_k * jnp.outer(yk, jnp.transpose(sk))) + rho_k * jnp.outer(sk, jnp.transpose(sk))
        return Hk_prime

    def quasi_newton_solve(self, objective_fn, x0):
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        xk = x0 # initial point
        dfdx_0 = objective_fn.dfdx(xk) # gradient at current point
        gradient_eval += 1 # gradient evaluation counter
        dfdx_k = dfdx_0
        dfdx_k_prime = dfdx_0
        Hk = jnp.eye(xk.shape[0]) # initial inverse Hessian approximation
        iter = 0 # iteration counter
        fx_traj = [objective_fn.f(xk)] # fx path
        dfdx_traj = [jnp.linalg.norm(dfdx_k)] # gradient path

        while (not self.check_converge(dfdx_k, dfdx_0, iter)) and (iter <= self.max_iterations - 1):  # check convergence or max iteration
            # search direction
            dfdx_k = deepcopy(dfdx_k_prime)
            pk = -Hk @ dfdx_k

            # step length
            alpha_k, func_eval_num, gradient_eval_num = self.line_searcher.search_stepsize(objective_fn, xk, pk, 1.0) 
            function_eval += func_eval_num # function evaluation counter
            gradient_eval += gradient_eval_num  # gradient evaluation counter
            
            # take a step
            prev_xk = deepcopy(xk)
            xk += alpha_k * pk
            dfdx_k_prime = objective_fn.dfdx(xk)
            gradient_eval += 1 # gradient evaluation counter

            # compute yk, sk for BFGS update
            sk = xk - prev_xk
            yk = dfdx_k_prime - dfdx_k
            # only perform BFGS udpate when yk.T @ sk is positive, skip otherwise
            if jnp.transpose(yk) @ sk > self.epsilon * jnp.linalg.norm(yk) * jnp.linalg.norm(sk):
                print("BFGS update performed")
                Hk = self.bfgs_update(Hk, sk, yk)

            print("Iter: ", iter, "f(x): ", objective_fn.f(xk), "||dfdx||: ", jnp.linalg.norm(dfdx_k), "alpha: ", alpha_k)
            fx_traj.append(objective_fn.f(xk))
            dfdx_traj.append(jnp.linalg.norm(dfdx_k))
            iter += 1

        if iter >= self.max_iterations:  # print info
            print("Newton solve exceeds max iter!")
        print("Function evaluation: ", function_eval, "Gradient evaluation: ", gradient_eval)

        return xk, objective_fn.f(xk), fx_traj, dfdx_traj