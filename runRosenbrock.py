import jax.numpy as jnp
import numpy as np
from src.objective_function import construct_problem
from src.meta_solver import MetaSolver, optSolverNewtonNinjas
from src.utils import plot_results

"""
This is a script file for running all 12 approaches on two Rosenbrock functions with dimensions 2 and 100.
"""

if __name__ == '__main__':

    # construct problems to be solved
    rng = np.random.default_rng(seed = 0) # random number generator
    function_name_list = ['Rosenbrock_2', 'Rosenbrock_100'] # problem name
    p8_starting_point = jnp.ones(100)
    p8_starting_point = p8_starting_point.at[0].set(-1.2)
    x0_list = [jnp.array([-1.2, 1.0]), p8_starting_point] # starting points
    problem_list = [construct_problem(function_name_list[ii], x0_list[ii]) for ii in np.arange(len(function_name_list))] # construct a list of problem objects
    
    # list of approach names
    approach_name_list = ["GradientDescent", "GradientDescentW", "ModifiedNewton", "ModifiedNewtonW", "NewtonCG", "NewtonCGW", "BFGS", "BFGSW", "DFP", "DFPW", "LBFGS", "LBFGSW"]
    
    # initialize two dictionaries to store the results for plotting
    fx_traj_dict = {key: {subkey: None for subkey in approach_name_list} for key in function_name_list}
    dfdx_traj_dict = {key: {subkey: None for subkey in approach_name_list} for key in function_name_list}

    for problem in problem_list:
        for approach in approach_name_list:
            xk, fx, fx_traj, dfdx_traj = optSolverNewtonNinjas(problem, approach, None) # solve the problem using the assigned approach
            fx_traj_dict[problem.name][approach] = fx_traj # store function value path along iterations
            dfdx_traj_dict[problem.name][approach] = dfdx_traj # store gradient norm path along iterations

    plot_results(fx_traj_dict, dfdx_traj_dict, function_name_list, approach_name_list) # plot the results






