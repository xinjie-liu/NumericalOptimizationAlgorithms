import jax.numpy as jnp
import numpy as np
from jax import jax, jit
from functools import partial
from timeit import timeit
from copy import deepcopy
from src.line_search import ArmijoLineSearch, WolfeLineSearch

"""
Implementation of L-BFGS Quasi Newton method
"""

class QuasiNewtonLBFGSSolver():
    def __init__(self, term_tol = 10**-6, max_iterations = 1000, line_searcher = "ArmijoLineSearch", memory_size = 10):
        self.term_tol = term_tol # termination tolerance
        self.max_iterations = max_iterations # max iterations
        if line_searcher == "ArmijoLineSearch": # choose line search method
            self.line_searcher = ArmijoLineSearch()
        elif line_searcher == "WolfeLineSearch":
            self.line_searcher = WolfeLineSearch()
        self.memory_size = memory_size if memory_size >= 1 else 10 # memory size

    def set_solver_name(self, name):
        self.name = name  # set name

    def check_converge(self, dfdx_k, dfdx_0, iter): # check gradient norm for termination of Newton iterations
        return jnp.linalg.norm(dfdx_k) <= (self.term_tol) * max(1, jnp.linalg.norm(dfdx_0)) 

    def l_bfgs_two_loop_recursion(self, k, m, Hk0, dfdx_k, sk_lst, yk_lst): 
        """
        Two loop recursion for updating Hessian inverse approximation and computing Newton's direction
        """
        q = dfdx_k
        alpha_lst = []
        for ii in range(len(sk_lst)-1, -1, -1):
            rho_i = 1 / (yk_lst[ii] @ sk_lst[ii] + 1e-8)
            alpha_i = rho_i * sk_lst[ii] @ q
            alpha_lst.insert(0, alpha_i)
            q = q - alpha_i * yk_lst[ii]
        r = Hk0 @ q
        for ii in range(0, len(sk_lst)):
            rho_i = 1 / (yk_lst[ii] @ sk_lst[ii] + 1e-8)
            beta = rho_i * yk_lst[ii] @ r
            r = r + sk_lst[ii] * (alpha_lst[ii] - beta)
        return r

    def quasi_newton_solve(self, objective_fn, x0):
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        xk = x0 # initial point
        dfdx_0 = objective_fn.dfdx(xk) # gradient at current point
        dfdx_k = dfdx_0
        dfdx_k_prime = dfdx_0
        gradient_eval += 1 # gradient evaluation counter
        m = min(xk.shape[0], self.memory_size) # memory size
        sk_lst = [] # past sk
        yk_lst = [] # past yk
        iter = 0 # iteration counter
        max_iter = self.max_iterations # max iterations
        fx_traj = [objective_fn.f(xk)] # fx path
        dfdx_traj = [jnp.linalg.norm(dfdx_k)] # gradient path

        while (not self.check_converge(dfdx_k, dfdx_0, iter)) and (iter <= max_iter - 1):  # check convergence or max iteration
            # initial inverse Hessian approximation
            gamma_k = 1 if iter == 0 else (sk_lst[-1] @ yk_lst[-1] + 1e-6) / (yk_lst[-1] @ yk_lst[-1] + 10**-6)
            Hk0 = gamma_k * jnp.eye(xk.shape[0]) 
            
            # search direction
            dfdx_k = deepcopy(dfdx_k_prime)
            pk = -self.l_bfgs_two_loop_recursion(iter, m, Hk0, dfdx_k, sk_lst, yk_lst)

            # step length
            alpha_k, func_eval_num, gradient_eval_num  = self.line_searcher.search_stepsize(objective_fn, xk, pk, 1.0) 
            function_eval += func_eval_num # function evaluation counter
            gradient_eval += gradient_eval_num  # gradient evaluation counter
            
            # take a step
            prev_xk = deepcopy(xk)
            xk += alpha_k * pk
            dfdx_k_prime = objective_fn.dfdx(xk)
            gradient_eval += 1  # gradient evaluation counter

            if iter > m: # discard old sk, yk
                sk_lst.pop(0)
                yk_lst.pop(0)

            # compute yk, sk for BFGS update
            sk = xk - prev_xk
            yk = dfdx_k_prime - dfdx_k
            sk_lst.append(sk)
            yk_lst.append(yk)

            print("Iter: ", iter, "f(x): ", objective_fn.f(xk), "||dfdx||: ", jnp.linalg.norm(dfdx_k), "alpha: ", alpha_k)
            fx_traj.append(objective_fn.f(xk))
            dfdx_traj.append(jnp.linalg.norm(dfdx_k))
            iter += 1

        if iter >= max_iter: # print info
            print("Newton solve exceeds max iter!")
        print("Function evaluation: ", function_eval, "Gradient evaluation: ", gradient_eval) 

        return xk, objective_fn.f(xk), fx_traj, dfdx_traj