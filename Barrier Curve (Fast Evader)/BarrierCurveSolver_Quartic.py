import numpy as np
import matplotlib.pyplot as plt
# See weburl https://www.wolframalpha.com/input?i2d=true&i=Power%5Bt%2C4%5D+-+4+Power%5Bt%2C3%5D+%2B+2+%5C%2840%291+-+Power%5BSubscript%5Bx%2C+E%5D%2C2%5D+-+3+Power%5BSubscript%5By%2C+E%5D%2C2%5D+%2B+Power%5BSubscript%5Bx%2C+P%5D%2C2%5D%5C%2841%29+Power%5Bt%2C2%5D+%2B4+%5C%2840%29Power%5BSubscript%5Bx%2C+E%5D%2C2%5D+-+Power%5BSubscript%5By%2C+E%5D%2C2%5D+-+Power%5BSubscript%5Bx%2C+P%5D%2C2%5D+%2B+1%5C%2841%29+t+%2B+Power%5B%5C%2840%29Power%5BSubscript%5Bx%2C+E%5D%2C2%5D+%2B+Power%5BSubscript%5By%2C+E%5D%2C2%5D+-+Power%5BSubscript%5Bx%2C+P%5D%2C2%5D+%2B+1%5C%2841%29%2C2%5D+%2B+4+%5C%2840%29Power%5BSubscript%5Bx%2C+P%5D%2C2%5D+-+1%5C%2841%29+Power%5BSubscript%5By%2C+E%5D%2C2%5D+%3D%3D+0%0A%0A%0A
# Function to compute the quartic equation coefficients
def quartic_coeffs(x_P, x_E, y_E):
    # Coefficients of the quartic equation
    a4 = 1  # t^4 term
    a3 = -4  # t^3 term
    a2 = 2 * (1 - x_E**2 - 3*y_E**2 + x_P**2)  # t^2 term
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)  # t^1 term
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2  # constant term
    
    return [a4, a3, a2, a1, a0]

def solve_quartic(x_P, x_E):
    y_values = np.linspace(0.0, 1, 10)  # Scan y_E from 0.0 to 1.0
    max_y_E = -np.inf  # Initialize with negative infinity
    print("Solving for y_E of : ")
    for y_E in y_values:
        print(f'y_E: {y_E}')
        coeffs = quartic_coeffs(x_P, x_E, y_E)
        roots = np.roots(coeffs)
        print(f'Found Roots: {roots} ')
        real_roots = [r for r in roots if np.isreal(r) and r >= 0]  # Only positive real roots
        print(f'Found soln: {real_roots}')
        if real_roots and y_E > max_y_E:  # Use '>' for proper comparison
            max_y_E = y_E  # Update max_y_E when a valid solution is found
    
    if max_y_E == -np.inf:
        return None  # If no valid y_E was found, return None
    return max_y_E  # Return the highest y_E with real roots

# Example parameters
x_P = 1.2  # example value for x_P

# Scan x_E from 0 to x_P
x_E_values = np.linspace(0.0, x_P, 100)
highest_y_E_values = []

for x_E in x_E_values:
    print(f'X_E: {x_E}')
    max_y_E = solve_quartic(x_P, x_E)
    if max_y_E is not None:
        highest_y_E_values.append(max_y_E)
    else:
        highest_y_E_values.append(np.nan)  # In case no real roots are found

# Now plot the highest y_E for each x_E
plt.figure(figsize=(10, 10))

# Plot the original curve
plt.scatter(x_E_values, highest_y_E_values,s = 5, color='red' )

# Mirror across the y-axis (multiply x_E by -1)
plt.scatter(-x_E_values, highest_y_E_values,s = 5, color='red')

# Mirror across the x-axis (multiply y_E by -1)
plt.scatter(x_E_values, -np.array(highest_y_E_values), s = 5,color='red')

# Mirror across both x and y axes (multiply both x_E and y_E by -1)
plt.scatter(-x_E_values, -np.array(highest_y_E_values), s = 5,color='red')

# Plot the first pursuer's position at (x_P, 0)
plt.scatter([x_P], [0], color='blue')

# Draw the capture radius for the first pursuer at (x_P, 0)
capture_radius_1 = plt.Circle((x_P, 0), 1, color='blue', fill = False)
plt.gca().add_patch(capture_radius_1)

# Plot the second pursuer's position at (-x_P, 0)
plt.scatter([-x_P], [0], color='blue')

# Draw the capture radius for the second pursuer at (-x_P, 0)
capture_radius_2 = plt.Circle((-x_P, 0), 1, color='blue', fill=False )
plt.gca().add_patch(capture_radius_2)

# Plot settings
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.xlabel('$x_E$')
plt.ylabel('$y_E$')
plt.title('Capturability Boundary with Reflections and Capture Radii of Two Pursuers')
plt.grid(False)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
