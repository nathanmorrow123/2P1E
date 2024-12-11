import numpy as np
import co_class as co
import matplotlib.pyplot as plt

"""
    runner file to generate cartesian oval plots for the 2P1E fast evader DG with capture circles
    nathan morrow 11 dec 2024

"""

# Show the plot
def main():
    # Define conditions
    mu = np.sqrt(2) # Speed ratio, Ve/Vp
    x_p = np.sqrt(2) # Starting Coord for Pursurer
    x_e = 0  # Starting Coord for Evader
    
    # Set Figure Size
    plt.figure(figsize=(16, 8))
    # Set the aspect of the plot to be equal
    
    plt.style.use('dark_background')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True  # Enable LaTeX rendering
    co.plotCartesianOval(x_p,x_e,mu)

    #plt.gca().set_aspect('equal', adjustable='box')
    # Labels and title
    plt.xlabel(r'$x_E$', fontsize = 20 )
    plt.ylabel(r'$y_E$', fontsize = 20 )
    plt.title(r"2P1E Cartesian $\textit{oval}$", fontsize = 24)
    #plt.legend()
    plt.grid(True, alpha = 0.3)
    #plt.set_cmap('hot')
    plt.savefig('Results/CO_2P1E_XE_ITER.png',dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
