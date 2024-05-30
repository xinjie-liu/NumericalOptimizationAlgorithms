import matplotlib.pyplot as plt
import numpy as np


"""
This file defines function for plotting
"""

def plot_results(fx_traj_dict, dfdx_traj_dict, function_name_list, approach_name_list):
    # Define line styles and colors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']
    colors = plt.cm.tab10.colors  # Use a color map for distinct colors
    for problem_name in function_name_list:
        # Create a figure and two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # subfigure one for plots of function value vs. iterations
        for ii in np.arange(len(approach_name_list)):
            approach_name = approach_name_list[ii]
            fx_traj = fx_traj_dict[problem_name][approach_name]
            ax1.plot(np.arange(len(fx_traj)) + 1, fx_traj, label=approach_name, linestyle=line_styles[ii % len(line_styles)], marker=markers[ii % len(markers)], color=colors[ii % len(colors)])
        ax1.set_title('Function value f(xk)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Iteration', fontsize=14)
        ax1.set_ylabel('f(xk)', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # subfigure two for plots of function gradient norm vs. iterations
        for ii in np.arange(len(approach_name_list)):
            approach_name = approach_name_list[ii]
            dfdx_traj = dfdx_traj_dict[problem_name][approach_name]
            ax2.plot(np.arange(len(dfdx_traj)) + 1, dfdx_traj, label=approach_name, linestyle=line_styles[ii % len(line_styles)], marker=markers[ii % len(markers)], color=colors[ii % len(colors)])
        ax2.set_title('Gradient norm ||df/dxk||_2', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Iteration', fontsize=14)
        ax2.set_ylabel('||df/dxk||_2', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    
        # Add gridlines
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax2.grid(True, linestyle='--', linewidth=0.5)

        # Add background color
        ax1.set_facecolor('#f0f0f0')
        ax2.set_facecolor('#f0f0f0')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Add title for the entire plot
        fig.suptitle(problem_name, fontsize=18, fontweight='bold')
        # Adjust layout to prevent overlap
        plt.subplots_adjust(top=0.9)  # Increase top spacing

        # Save the figure
        plt.savefig(problem_name + ".png")

        # Show the plot
        plt.show()