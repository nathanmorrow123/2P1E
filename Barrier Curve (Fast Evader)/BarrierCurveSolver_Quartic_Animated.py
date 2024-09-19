import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to compute the quartic equation coefficients
def quartic_coeffs(x_P, x_E, y_E):
    a4 = 1  # t^4 term
    a3 = -4  # t^3 term
    a2 = 2 * (1 - x_E**2 - 3*y_E**2 + x_P**2)  # t^2 term
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)  # t^1 term
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2  # constant term
    return [a4, a3, a2, a1, a0]

# Function to solve the quartic equation for a given x_P and x_E, scanning y_E
def solve_quartic(x_P, x_E):
    y_values = np.linspace(0.0, 1, 100)  # Scan y_E from 0.0 to 1.0
    max_y_E = -np.inf  # Initialize with negative infinity
    for y_E in y_values:
        coeffs = quartic_coeffs(x_P, x_E, y_E)
        roots = np.roots(coeffs)
        real_roots = [r for r in roots if np.isreal(r) and r >= 0]  # Only positive real roots
        if real_roots and y_E > max_y_E:  # Update max_y_E when a valid solution is found
            max_y_E = y_E
    if max_y_E == -np.inf:
        return None  # No valid y_E was found
    return max_y_E  # Return the highest y_E with real roots

# Initialize animation parameters
x_P_values = np.linspace(1.2, 5, 240)  # 240 frames of x_P varying from 1.2 to 2.5

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Setup axis properties
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('$x_E$')
ax.set_ylabel('$y_E$')
ax.set_title('Capturability Boundary Animation')
ax.set_aspect('equal', adjustable='box')

# Create scatter plots for original and mirrored points, and pursuer circles
scatter_main, = ax.plot([], [], 'ro', ms=5)  # Main curve
scatter_mirror_x, = ax.plot([], [], 'ro', ms=5)  # Mirror across x-axis
scatter_mirror_y, = ax.plot([], [], 'ro', ms=5)  # Mirror across y-axis
scatter_mirror_xy, = ax.plot([], [], 'ro', ms=5)  # Mirror across both axes
pursuer_circle_1 = plt.Circle((0, 0), 1, color='blue', fill=False)  # First pursuer
pursuer_circle_2 = plt.Circle((0, 0), 1, color='blue', fill=False)  # Second pursuer
ax.add_patch(pursuer_circle_1)
ax.add_patch(pursuer_circle_2)

# Update function for each frame of the animation
def update(frame):
    x_P = x_P_values[frame]  # Get the current x_P value for this frame
    highest_y_E_values = []
    x_E_values = np.linspace(0, x_P, 100)  # 240 frames of x_P varying from 1.2 to 2.5
    # Solve for each x_E in the range
    for x_E in x_E_values:
        max_y_E = solve_quartic(x_P, x_E)
        if max_y_E is not None:
            highest_y_E_values.append(max_y_E)
        else:
            highest_y_E_values.append(np.nan)  # No solution for this x_E

    # Update the main curve and its mirrored versions
    highest_y_E_values = np.array(highest_y_E_values)
    scatter_main.set_data(x_E_values, highest_y_E_values)
    scatter_mirror_x.set_data(x_E_values, -highest_y_E_values)
    scatter_mirror_y.set_data(-x_E_values, highest_y_E_values)
    scatter_mirror_xy.set_data(-x_E_values, -highest_y_E_values)

    # Update the positions of the pursuer circles
    pursuer_circle_1.center = (x_P, 0)
    pursuer_circle_2.center = (-x_P, 0)

    # Adjust the plot limits dynamically
    x_margin = 0.2  # Add a margin around x_P
    y_max = max(np.nanmax(highest_y_E_values), 1) + 0.5  # Add a margin to the y-axis
    
    # Set x and y limits dynamically
    ax.set_xlim([-x_P - x_margin, x_P + x_margin])
    ax.set_ylim([-y_max, y_max])

    return scatter_main, scatter_mirror_x, scatter_mirror_y, scatter_mirror_xy, pursuer_circle_1, pursuer_circle_2

# Create the animation
anim = FuncAnimation(fig, update, frames=240, interval=1000/30, blit=True)

# Save the animation as a video file (optional)
anim.save('capturability_animation.gif', fps=30)

# Display the animation
plt.show()
