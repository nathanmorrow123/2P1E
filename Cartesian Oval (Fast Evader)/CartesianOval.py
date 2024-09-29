import numpy as np
import matplotlib.pyplot as plt


def plotCartesianOval(x_p,x_e,mu):
    # Args ( Pursurer starting x coord, Evader starting coord, Speed ratio Ve/Vp)        
    t_min = (1/(1+mu))*(x_p-x_e-1)
    t_max = (1/np.sqrt(mu**2-1))*(np.sqrt((x_p-x_e)**2-1))
    numerator = np.sqrt((-2 * x_p + 2 * x_e - 2)**2 - 4 * (mu**4 - 2 * mu**2 + 1) * (2 * x_e * x_p - x_p**2 - 2 * x_p - x_e**2 + 2 * x_e - 1)) + 2 * x_p - 2 * x_e + 2
    denominator = 2 * (mu**4 - 2 * mu**2 + 1)
    t_max_true = numerator/denominator

    # Generate parameter t
    t = np.linspace(t_min, t_max_true, 1000, endpoint=True)

    # Parametric equations
    x1 = [x_p+((mu**2-1)*t**2-2*t-(x_p-x_e)**2-1)/(2*(x_p-x_e)),x_p+((mu**2-1)*t**2-2*t-(x_p-x_e)**2-1)/(2*(x_p-x_e))]
    y1 = [np.sqrt((t+1)**2-(x1[0]-x_p)**2),-np.sqrt((t+1)**2-(x1[1]-x_p)**2)] # Two values for y a positive and a neqative sqrt
    x2 = [-x_p-((mu**2-1)*t**2-2*t-(x_p-x_e)**2-1)/(2*(x_p-x_e)),-x_p-((mu**2-1)*t**2-2*t-(x_p-x_e)**2-1)/(2*(x_p-x_e))]
    y2 = [np.sqrt((t+1)**2-(x2[0]+x_p)**2),-np.sqrt((t+1)**2-(x2[1]+x_p)**2)] # Two values for y a positive and a neqative sqrt

    # Plot the Cartesian oval
    plt.scatter(x1, y1, s=1, label='P1 Cartesian Oval')
    plt.scatter(x2, y2, s=1, label = 'P2 Cartesian Oval')
    # Plot the foci
    plt.plot(x_p, 0, 'o', label='(Pursuer 1)')
    plt.plot(-x_p, 0, 'o', label='(Pursuer 1)')
    plt.plot(x_e, 0,  '.', label ='(Evader)')

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
    for x_e in np.linspace(np.sqrt(2)-1,0,8,endpoint=False):
        plotCartesianOval(x_p,x_e,mu)

    #plt.gca().set_aspect('equal', adjustable='box')
    # Labels and title
    plt.xlabel('x', fontsize = 20 )
    plt.ylabel('y', fontsize = 20 )
    plt.title('Cartesian Oval for Pursuer-Evader Problem', fontsize = 24)
    #plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.set_cmap('hot')
    plt.savefig('CO_2P1E_XE_ITER.png',dpi=600)
    plt.show()

if __name__ == "__main__":
    main()

