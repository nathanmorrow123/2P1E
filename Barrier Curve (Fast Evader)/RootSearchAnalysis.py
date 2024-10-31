import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy.polynomial.polynomial import Polynomial
import pandas as pd

# Given a quartic equation that describes time to capture from the state variables x_P, x_E, y_E
states = [0, 0, 0]

def getCoeffs():
    # Define the quartic polynomial with the current state variables
    x_P, x_E, y_E = states
    a4 = 1
    a3 = -4
    a2 = 2 * (1 - x_E**2 - 3*y_E**2 + x_P**2)
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2
    
    # Solve for the roots of the quartic polynomial
    coeffs = [a0, a1, a2, a3, a4]
    return coeffs

def func(t):
    """
    Function to compute the quartic equation coefficients based on state variables x_P, x_E, y_E.
    """
    x_P, x_E, y_E = states
    a4 = 1  # t^4 term
    a3 = -4  # t^3 term
    a2 = 2 * (1 - x_E**2 - 3*y_E**2 + x_P**2)  # t^2 term
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)  # t^1 term
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2  # constant term
    
    # Return the quartic polynomial expression for root-finding
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4

# Function to remove outliers based on IQR
def remove_outliers(roots):
    """
    Remove outliers from the last positive root values in root_data using IQR method
    """
    roots = roots[np.isreal(roots)]  # Filter only real roots
    Q1 = np.percentile(roots, 5)
    Q3 = np.percentile(roots, 95)
    IQR = Q3 - Q1

    # Define the bounds for non-outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    filtered_data = roots[(roots >= lower_bound) & (roots <= upper_bound)]

    return filtered_data

# Main function to calculate and plot roots
def mirrorData(state_data, root_data):
    """
    Since time to capture function is symmetric across both x and y axes
    This function mirrors the root data
    """
    # Stack original data with mirrored data
    mirrorData = (np.array((state_data[:, 0],-1*state_data[:,1],state_data[:,2])).T)
    state_data = np.vstack((state_data, mirrorData))
    mirrorData = (np.array((state_data[:, 0],-1*state_data[:,1],1*state_data[:,2])).T)
    state_data = np.vstack((state_data, mirrorData))
    mirrorData = (np.array((state_data[:, 0],state_data[:,1],-1*state_data[:,2])).T)
    state_data = np.vstack((state_data, mirrorData))
    root_data = np.vstack((root_data, root_data))
    root_data = np.vstack((root_data, root_data))
    root_data = np.vstack((root_data, root_data))
    
    return state_data, root_data

def methodOne():
    """
    Solve for all real positive roots using the quartic polynomial coefficients.
    """
    coeffs = getCoeffs()
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    return real_roots if len(real_roots) == 4 else None

def methodTwo(roots_guess):
    roots = optimize.fsolve(func,roots_guess)
    if(any(np.isclose(func(roots),[0,0,0,0]))):
        return roots
    else:
        return None

def plotTimeSurface(state_data, root_data):
    """
    Creates a 3D scatter plot for x_E, y_E, and the last positive root.
    """
    state_data, root_data = mirrorData(state_data,root_data)
    x_P = state_data[:, 0]
    x_E = state_data[:, 1]
    y_E = state_data[:, 2]
    positive_min_roots = []
    for roots in root_data:
        roots = roots[roots>=0]
        for root in roots:
            if(root == min(roots)):
                positive_min_roots.append(root)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_E, y_E, root_data.min(axis=1) , c=root_data.min(axis=1), cmap='viridis')
    ax.set_xlabel('$x_E$')
    ax.set_ylabel('$y_E$')
    ax.set_zlabel('Last Positive Root')
    fig.colorbar(sc)
    plt.show()

def plotRootAnalysis(state_data, root_data):
    """
    Takes in root data and displays roots as y_E varies for a single x_E.
    """
    x_E = state_data[:, 1]
    x_E_Picked_i = np.where(x_E == np.median(x_E))
    y_E = state_data[x_E_Picked_i,2]
    root_data = root_data[x_E_Picked_i]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_E.T, root_data)
    ax.set_xlabel('y_E')
    ax.set_ylabel('Roots')
    plt.show()

def displayRootTable(state_data, root_data):
    """
    Display the root data in a tabular format, formatted to 3 decimal points.
    """
    df = pd.DataFrame({
        'x_P': state_data[:, 0],
        'x_E': state_data[:, 1],
        'y_E': state_data[:, 2],
        'Roots': [", ".join(f"{r:.3f}" for r in roots) for roots in root_data]
    })
    print(df)

def main():
    x_P = 1.2
    x_E_values = np.linspace(0.0, x_P, 100)
    y_E_values = np.linspace(0.0, 1.0, 100)

    # Store roots for plotting
    root_data = []
    state_data = []
    states = [x_P,x_E_values[0],y_E_values[0]]
    x_P,x_E,y_E = states
    roots_guess =  [1 - np.sqrt( -x_P**2 + x_E**2 + 2),1 + np.sqrt( -x_P**2 + x_E**2 + 2),1 - np.sqrt( -x_P**2 + x_E**2 + 2),1 + np.sqrt( -x_P**2 + x_E**2 + 2)]
    roots_guess = [0,0,0,0]
    states[0] = x_P
    for x_E in x_E_values:
        states[1] = x_E
        for y_E in y_E_values:
            states[2] = y_E
            roots = [1 - np.sqrt( -x_P**2 + x_E**2 + 2),1 + np.sqrt( -x_P**2 + x_E**2 + 2), 1 - np.sqrt( -x_P**2 + x_E**2 + 2),1 + np.sqrt( -x_P**2 + x_E**2 + 2)] if y_E == 0 else methodTwo(roots_guess)
            if roots is not None and any(roots) > 0:
                root_data.append(roots)
                state_data.append(states.copy())

    state_data = np.array(state_data)
    root_data = np.array(root_data)
    plotRootAnalysis(state_data, root_data)
    plotTimeSurface(state_data, root_data)
    displayRootTable(state_data, root_data)

if __name__ == '__main__':
    main()
