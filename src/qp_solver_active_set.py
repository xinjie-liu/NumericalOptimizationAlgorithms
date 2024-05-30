import jax.numpy as jnp
import numpy as np
import scipy
from jax import jax, jit
from functools import partial
from timeit import timeit
import scipy.linalg
import sys

class QPSolverActiveSet():
    def __init__(self, G, c, A, b):
        # TODO: assertion that only allows PSD and symmetric metrices G
        self.G = G
        self.c = c
        self.A = A
        self.b = b
        self.max_iter = 100

    def qp_initialize(self, G, c, A, b):
        self.G = G
        self.c = c
        self.A = A
        self.b = b

    def initialize_working_set(self, x0):
        working_set = jnp.where(abs(self.A @ x0 - self.b) <= 1e-5)[0]
        return working_set

    def qp_solve(self, x0):
        if not all(self.A @ x0 >= self.b):
            raise Exception("Initial point of the QP is infeasible!")
        self.working_set = self.initialize_working_set(x0)

        k = 0
        xk = x0
        solution_trace = [x0]
        while k <= self.max_iter:
            c = self.G @ xk + self.c
            if jnp.shape(self.working_set)[0] == 0: # no constraints are active
                A = None
                b = None
            else:
                A = self.A[self.working_set, :]
                b = jnp.zeros(self.working_set.shape[0])
            G = self.G
            if np.all(np.linalg.eigvals(G) > 0) and jnp.shape(self.working_set)[0] > 0:
                pk, lambda_k = self.eqp_solve(G, c, A, b, eqp_solver = 'schur_complement')
            else:
                pk, lambda_k = self.eqp_solve(G, c, A, b, eqp_solver = 'direct_solve_kkt')
            lambda_k_ = self.compute_langrangian_multipliers(lambda_k) # compute duals for the QP iteration
            if np.linalg.norm(pk) <= 1e-5: # xk is optimum for the subproblem EQP
                if (lambda_k is np.nan) or ((not lambda_k is np.nan) and all(lambda_k >= 0)): # solution found
                    x_star = xk
                    lambda_star = self.compute_langrangian_multipliers(lambda_k)
                    print("QP solve converged!")
                    self.print_itr_info(k, xk, lambda_star, 0.0, pk)
                    return x_star, lambda_star, solution_trace
                elif (not lambda_k is np.nan) and (not all(lambda_k >= 0)):
                    self.working_set = jnp.delete(self.working_set, jnp.argmin(lambda_k))
                    xk = xk
                    alpha_k = 0.0
            else:
                if all(self.A @ (xk + pk) >= self.b): # if xk + pk feasible
                    xk += pk # a big step with alpha = 1
                    alpha_k = 1.0
                else:
                    alpha_lst = []
                    for ii in np.arange(self.A.shape[0]):
                        if ii in self.working_set:
                            alpha_lst = np.append(alpha_lst, 1.0)
                        if not ii in self.working_set:
                            if self.A[ii] @ pk >= 0:
                                alpha_lst = np.append(alpha_lst, 1.0)
                            else:
                                alpha_lst = np.append(alpha_lst, (self.b[ii] - self.A[ii] @ xk) / (self.A[ii] @ pk))
                    alpha_k = np.min(alpha_lst)
                    # adding the blocking constraint (if any) to the working set
                    if not np.argmin(alpha_lst) in self.working_set: 
                        self.working_set = np.insert(self.working_set, 0, np.argmin(alpha_lst))
                        self.working_set = np.sort(self.working_set)
                    xk += alpha_k * pk

            solution_trace.append(xk)
            self.print_itr_info(k, xk, lambda_k_, alpha_k, pk)
            k += 1

        print("Maximum QP iteration reached! The returned solution might be inaccurate!")
        x_star = xk
        lambda_star = self.compute_langrangian_multipliers(lambda_k)
        return x_star, lambda_star, solution_trace

    def print_itr_info(self, k, xk, lambda_k, alpha_k, pk):
        print("Iteration: ", k)
        print("f(xk): ", 0.5 * xk.T @ self.G @ xk + xk.T @ self.c)
        print("xk: ", xk)
        print("lambda_k: ", lambda_k)
        print("working set: ", self.working_set)
        print("alpha_k: ", alpha_k)
        print("||pk||_2: ", np.linalg.norm(pk))
        print("======================================")

    def compute_langrangian_multipliers(self, lambda_k):
        if lambda_k is np.nan:
            lambda_k_reconstruct = jnp.zeros(self.A.shape[0])
        else:
            lambda_k_reconstruct = jnp.zeros(self.A.shape[0])
            lambda_k_reconstruct = lambda_k_reconstruct.at[self.working_set].set(lambda_k)
        return lambda_k_reconstruct

    def eqp_solve(self, G, c, A, b, eqp_solver = 'schur_complement'):
        dim_n = jnp.shape(G)[1]
        dim_m = jnp.shape(A)[0] if not A == None else 0
        # KKT system
        LHS_upper = jnp.concatenate((G, A.T), axis = 1) if not dim_m == 0 else G
        LHS_lower = jnp.concatenate((A, jnp.zeros((dim_m, dim_m))), axis = 1) if not dim_m == 0 else None
        
        LHS = jnp.concatenate((LHS_upper, LHS_lower)) if not LHS_lower == None else LHS_upper
        RHS = jnp.concatenate((-c, b)) if not b == None else -c

        if eqp_solver == 'schur_complement':
            if not np.all(np.linalg.eigvals(G) > 0):
                raise Exception("Matrix G not positive definite, cannot perform Schur-Complement method!")
            else:
                print("Schur complement solve performed!")
                lambda_star = scipy.linalg.solve(A @ np.linalg.inv(G) @ A.T, b + A @ np.linalg.inv(G) @ c)
                x_star = scipy.linalg.solve(G, A.T @ lambda_star - c)
                return x_star, lambda_star
        elif eqp_solver == 'direct_solve_kkt':            
            try:
                sol = scipy.linalg.solve(LHS, RHS, assume_a = 'sym')
            except np.linalg.LinAlgError:
                print("Warning: Inner EQP iteration infeasible: KKT matrix singular!")
            except np.linalg.LinAlgWarning:
                print("Warning: Inner EQP iteration ill-conditioned!")
            except ValueError:
                print("Warning: Inner EQP iteration infeasible: Size mismatches detected or KKT matrix not square!")
            except:
                print("Warning: Inner EQP iteration not cleanly solved!")
            else:
                x_star = sol[0:dim_n]
                lambda_star = -sol[dim_n:] if not dim_m == 0 else jnp.nan
                return x_star, lambda_star
            return jnp.nan, jnp.nan

    