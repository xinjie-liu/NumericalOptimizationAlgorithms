import jax.numpy as jnp
import numpy as np
from jax import jax, jit
from functools import partial
from copy import deepcopy

"""
This file contains implementation of two line search approaches 
"""

"""
An implementation of Armijo line search algorithm.
"""
class ArmijoLineSearch():
    def __init__(self,
        alpha_init = 1,
        c1 = 10**-4,
        tau = 0.5):
        """
        If being used along with jit, the internal parameter values should keep fixed under current implementation, 
        otherwise, undesired behavior may occur.
        """
        self.alpha_init = alpha_init # initial step size
        self.c1 = c1 # c1 parameter
        self.tau = 0.5 # step size shrinking factor for each iteration

    def checkArmijo_condition(self, objective_fn, dfdx, xk, pk, alpha_t): # Check Armijo condition
        return objective_fn.f(xk + alpha_t * pk) <= objective_fn.f(xk) + self.c1 * alpha_t * dfdx @ pk 

    def search_stepsize(self, objective_fn, xk, pk, alpha_t):
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        dfdx = objective_fn.dfdx(xk) # get function gradient at current point; note: can be optimized to use the gradient computed in the direction search step
        gradient_eval += 1 # gradient evaluation counter
        iter = 0 # iteration counter
        max_iter = 1000
        while True:
            # if Armijo condition is satisfied or the step size is too small, terminate line search and return the step size alpha_t
            if self.checkArmijo_condition(objective_fn, dfdx, xk, pk, alpha_t) or alpha_t <= 1e-8:
                function_eval += 2 # function evaluation counter
                # print("Armijo line search terminates nicely!")
                break
            # print(alpha_t) # TODO: verbose argument
            alpha_t *= self.tau # shrinking the step size if Armijo condition is not satisfied
            iter += 1 # iteration counter
        
        if iter >= max_iter: # print info if line search took too many steps
            print("Armijo line search exceeds max iter!")
        
        return alpha_t, function_eval, gradient_eval
    
"""
An implementation of Wolfe line search algorithm.
"""
class WolfeLineSearch():
    def __init__(self,
        alpha_init = 1,
        c1 = 10**-4,
        c2 = 0.9):
        """
        If being used along with jit, the internal parameter values should keep fixed under current implementation, 
        otherwise, undesired behavior may occur.
        """
        self.alpha_init = alpha_init # initial step size 
        self.c1 = c1 # c1 parameter
        self.c2 = c2 # c2 parameter

    def checkArmijo_condition(self, objective_fn, dfdx, xk, pk, alpha_t):  # Check Armijo condition
        return objective_fn.f(xk + alpha_t * pk) <= objective_fn.f(xk) + self.c1 * alpha_t * dfdx @ pk
    
    def checkCurvature_condition(self, objective_fn, dfdx, xk, pk, alpha_t): # Check Curvature condition
        return objective_fn.dfdx(xk + alpha_t * pk) @ pk >= self.c2 * dfdx @ pk

    def search_stepsize(self, objective_fn, xk, pk, alpha_t):
        function_eval = 0 # function evaluation counter
        gradient_eval = 0 # gradient evaluation counter
        dfdx = objective_fn.dfdx(xk) # get function gradient at current point; note: can be optimized to use the gradient computed in the direction search step
        gradient_eval += 1 # gradient evaluation counter
        alpha_l = 0 # lower bound for step size
        alpha_u = np.inf # upper bound for step size
        prev_alpha = alpha_t # step size computed from the previous iteration
        counter = 0 # counter: how many time steps the step size hasn't been updated
        iter = 0 # iteration counter
        max_iter = 1000 # maximum number of iteration
        while iter <= max_iter - 1: 
            if not self.checkArmijo_condition(objective_fn, dfdx, xk, pk, alpha_t): # armijo condition not satisfied
                alpha_u = alpha_t
                function_eval += 2
            else:
                if not self.checkCurvature_condition(objective_fn, dfdx, xk, pk, alpha_t): # curvature condition not satisfied
                    alpha_l = alpha_t
                    gradient_eval += 1
                else:
                    # If both the Armijo and the Curvature conditions are satisfied, terminate and return the step size
                    # print("Wolfe line search terminates nicely!")
                    break
            if alpha_u < np.inf:
                alpha_t = (alpha_l + alpha_u) / 2 # update step size, if upper bound finite
            else:
                alpha_t *= 2 # update step size, if upper bound infinite
            if alpha_t <= 1e-6: # terminate if the step size gets too small
                break
            if abs(alpha_t - prev_alpha) <= 1e-8: # if the step size hasn't been updated for 20 iterations, terminate (this is for bad situations where the upper and lower bounds coincide and the resulting step size does not satisfy termination conditions)
                counter += 1
                if counter >= 20:
                    break
            prev_alpha = deepcopy(alpha_t) # update previous step size
            iter += 1
        if iter >= max_iter: # print info if line search took too many steps
            print("Wolfe line search exceeds max iter!")
        return alpha_t, function_eval, gradient_eval