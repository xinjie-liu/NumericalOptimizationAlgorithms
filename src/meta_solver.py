from src.line_search import ArmijoLineSearch, WolfeLineSearch
from src.solver_steepest_descent import SteepestDescent
from src.solver_newton import NewtonSolver
from src.solver_newton_cg import NewtonConjugateGradientSolver
from src.solver_quasi_newton_bfgs import QuasiNewtonBFGSSolver
from src.solver_quasi_newton_dfp import QuasiNewtonDFPSolver
from src.solver_quasi_newton_l_bfgs import QuasiNewtonLBFGSSolver
import time

"""
This file contains a `MetaSolver` class and an `optSolverNewtonNinjas` function. They are used to initialize and call the solvers specified by the user in the required form.
"""

class MetaSolver():
    """
    Meta solver that constructs different solvers based on the user's input
    Inputs: approach name, options; if options = None, initialize the solver using default options
    After initialization, the class contains a "internal" solver that satisfies the user's instruction. 
    """
    def __init__(self, solver_name, options):
        if solver_name == "GradientDescent":
            if options == None: # initialize using default options
                self.solver = SteepestDescent()
            else:
                for argument in list(options.keys()): # initialize using user specified options
                    if not argument in ["term_tol", "max_iterations"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = SteepestDescent(**options)
        elif solver_name == "GradientDescentW":
            if options == None:
                self.solver = SteepestDescent(line_searcher="WolfeLineSearch")
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = SteepestDescent(**options, line_searcher = "WolfeLineSearch")
        elif solver_name == "ModifiedNewton":
            if options == None:
                self.solver = NewtonSolver(modify_hessian_flag = True)
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "beta"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = NewtonSolver(**options, modify_hessian_flag = True)
        elif solver_name == "ModifiedNewtonW":
            if options == None:
                self.solver = NewtonSolver(modify_hessian_flag = True, line_searcher="WolfeLineSearch")
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "beta"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = NewtonSolver(**options, modify_hessian_flag = True, line_searcher = "WolfeLineSearch")
        elif solver_name == "NewtonCG":
            if options == None:
                self.solver = NewtonConjugateGradientSolver()
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "eta_cg_tol"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = NewtonConjugateGradientSolver(**options)
        elif solver_name == "NewtonCGW":
            if options == None:
                self.solver = NewtonConjugateGradientSolver(line_searcher="WolfeLineSearch")
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "eta_cg_tol"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = NewtonConjugateGradientSolver(**options, line_searcher = "WolfeLineSearch")
        elif solver_name == "BFGS":
            if options == None:
                self.solver = QuasiNewtonBFGSSolver()
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "epsilon"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = QuasiNewtonBFGSSolver(**options)
        elif solver_name == "BFGSW":
            if options == None:
                self.solver = QuasiNewtonBFGSSolver(line_searcher="WolfeLineSearch")
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "epsilon"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = QuasiNewtonBFGSSolver(**options, line_searcher = "WolfeLineSearch")
        elif solver_name == "DFP":
            if options == None:
                self.solver = QuasiNewtonDFPSolver()
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "epsilon"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = QuasiNewtonDFPSolver(**options)
        elif solver_name == "DFPW":
            if options == None:
                self.solver = QuasiNewtonDFPSolver(line_searcher="WolfeLineSearch")
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "epsilon"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = QuasiNewtonDFPSolver(**options, line_searcher = "WolfeLineSearch")
        elif solver_name == "LBFGS":
            if options == None:
                self.solver = QuasiNewtonLBFGSSolver()
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "memory_size"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = QuasiNewtonLBFGSSolver(**options)
        elif solver_name == "LBFGSW":
            if options == None:
                self.solver = QuasiNewtonLBFGSSolver(line_searcher="WolfeLineSearch")
            else:
                for argument in list(options.keys()):
                    if not argument in ["term_tol", "max_iterations", "memory_size"]:
                        raise Exception("Please provide proper options excluding '{}' !".format(argument))
                self.solver = QuasiNewtonLBFGSSolver(**options, line_searcher="WolfeLineSearch")
        self.solver.set_solver_name(solver_name)
        
    def solve(self, problem):
        """
        Solve the given problem using the "internal" solver. 
        """
        if self.solver.name == "GradientDescent" or self.solver.name == "GradientDescentW":
            sol = self.solver.steepest_descent_solve(problem, problem.x0)
        elif self.solver.name == "ModifiedNewton" or self.solver.name == "ModifiedNewtonW":
            sol = self.solver.newton_solve(problem, problem.x0)
        elif self.solver.name == "NewtonCG" or self.solver.name == "NewtonCGW":
            sol = self.solver.newton_cg_solve(problem, problem.x0)
        elif self.solver.name == "BFGS" or self.solver.name == "BFGSW":
            sol = self.solver.quasi_newton_solve(problem, problem.x0)
        elif self.solver.name == "DFP" or self.solver.name == "DFPW":
            sol = self.solver.quasi_newton_solve(problem, problem.x0)
        elif self.solver.name == "LBFGS" or self.solver.name == "LBFGSW":
            sol = self.solver.quasi_newton_solve(problem, problem.x0)
        return sol

def optSolverNewtonNinjas(problem, method, options):
    """
    A sugar function that takes in problem, approach, and options and solves the problem using the assigned approach
    This is for satisfaction of the required solver calling form
    """
    print("#####################################################################################\n")
    print("#                                Problem {}                                         #\n".format(problem.name))
    print("#####################################################################################\n")
    print("                                                                                 \n")
    print("                                 {}                                         \n".format(method))
    print("                                                                                 \n")
    meta_solver = MetaSolver(method, options) # construct a meta solver with the user specified approach and options
    start = time.perf_counter()
    sol = meta_solver.solve(problem) # solve the problem
    end = time.perf_counter()
    elapsed = end - start # count the time used
    print(f'Time taken: {elapsed:.6f} seconds')
    
    return sol