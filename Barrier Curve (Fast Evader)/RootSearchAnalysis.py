import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy.polynomial.polynomial import Polynomial

# Given a quartic equation that describes time to capture from the state variables x_P, x_E, y_E
state_variables = [0, 0, 0]

def func(t):
    """
    Function to compute the quartic equation coefficients based on state variables x_P, x_E, y_E.
    """
    x_P, x_E, y_E = state_variables
    a4 = 1  # t^4 term
    a3 = -4  # t^3 term
    a2 = 2 * (1 - x_E**2 - 3*y_E**2 + x_P**2)  # t^2 term
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)  # t^1 term
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2  # constant term
    
    # Return the quartic polynomial expression for root-finding
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4

def find_last_positive_root():
    """
    Find the largest real positive root of the quartic equation for the current state variables.
    """
    # Define the quartic polynomial with the current state variables
    x_P, x_E, y_E = state_variables
    a4 = 1
    a3 = -4
    a2 = 2 * (1 - x_E**2 - 3*y_E**2 + x_P**2)
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2
    
    # Solve for the roots of the quartic polynomial
    coeffs = [a0, a1, a2, a3, a4]
    roots = np.roots(coeffs)
    
    # Keep only real positive roots
    real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    
    # If there are no positive real roots, return None
    if len(real_roots) == 0:
        return None
    
    # Return the largest positive real root
    return max(real_roots)
# Function to remove outliers based on IQR
def remove_outliers(root_data):
    """
    Remove outliers from the last positive root values in root_data using IQR method.
    root_data: np.array with columns [x_E, y_E, last_positive_root]
    """
    roots = root_data[:, 2]  # Extract the last positive root column

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(roots, 25)
    Q3 = np.percentile(roots, 75)
    IQR = Q3 - Q1

    # Define the bounds for non-outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    filtered_data = root_data[(roots >= lower_bound) & (roots <= upper_bound)]

    return filtered_data
# Main function to calculate and plot roots
def main():
    x_P_values = [1.1]
    x_E_values = np.linspace(0, 1, 100)
    y_E_values = np.linspace(0, 1, 100)

    # Store roots for plotting
    root_data = []

    for x_P in x_P_values:
        state_variables[0] = x_P
        for x_E in x_E_values:
            state_variables[1] = x_E
            for y_E in y_E_values:
                state_variables[2] = y_E
                last_positive_root = find_last_positive_root()
                print(f'State Variables: {state_variables}, Last Positive Root: {last_positive_root}')
                if last_positive_root is not None:
                    root_data.append((x_E, y_E, last_positive_root))
    
    # Convert root data into numpy arrays for plotting
    # Assuming root_data contains (x_E, y_E, root)

    # Convert root_data to numpy array for easier manipulation
    root_data = np.array(root_data)
    root_data = remove_outliers(root_data)
    # Create mirrored data across x and y axes
    mirrored_x_data = np.copy(root_data)
    mirrored_x_data[:, 0] = -mirrored_x_data[:, 0]  # Mirror across x-axis

    mirrored_y_data = np.copy(root_data)
    mirrored_y_data[:, 1] = -mirrored_y_data[:, 1]  # Mirror across y-axis

    mirrored_both_data = np.copy(root_data)
    mirrored_both_data[:, 0] = -mirrored_both_data[:, 0]  # Mirror x
    mirrored_both_data[:, 1] = -mirrored_both_data[:, 1]  # Mirror y

    # Stack original data with mirrored data
    all_data = np.vstack([root_data, mirrored_x_data, mirrored_y_data, mirrored_both_data])

    # Now all_data contains the original data and the mirrored versions across both axes

    x_E_plot, y_E_plot, root_plot = all_data[:,0], all_data[:,1], all_data[:,2]
    # Create 3D scatter plot for x_E, y_E, and the last positive root
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x_E_plot, y_E_plot, root_plot, c=root_plot, cmap='viridis')
    ax.set_xlabel('x_E')
    ax.set_ylabel('y_E')
    ax.set_zlabel('Last Positive Root')
    fig.colorbar(sc)
    plt.show()

if __name__ == '__main__':
    main()
