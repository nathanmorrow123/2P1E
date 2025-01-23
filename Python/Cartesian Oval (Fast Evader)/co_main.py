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
    mu = 1.1 # Speed ratio, Ve/Vp
    x_P = 1.1
    x_E = 0.05 
    y_E = 0.2

    co.plotCartesianOval(x_P,x_E,y_E,mu)

if __name__ == "__main__":
    main()
