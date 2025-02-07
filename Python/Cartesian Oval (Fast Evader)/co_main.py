import numpy as np
import co_class as co
import matplotlib.pyplot as plt

"""

    runner file to generate cartesian oval plots for the 2P1E fast evader DG with capture circles
    nathan morrow 11 dec 2024

"""

# Show the plot
def main():
    # Define States 
    mu = np.sqrt(2) # Speed ratio, Ve/Vp
    x_E = 0.1 
    y_E = 0.0
    eps = [0.02,0.04,0.1,0.2,0.3]
    x_P = mu / np.sqrt(mu**2 - 1)
    for e in eps:
        y_E = e
        co.plotCartesianOval(x_P,x_E,y_E,mu)
    
    

if __name__ == "__main__":
    main()
