import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Define constants
mu = np.sqrt(2)
x_p = np.sqrt(2)
x_e = 0.3

# Calculate t_min and t_max
t_min = (1 / (1 + mu)) * (x_p - x_e - 1)
t_max = (1 / np.sqrt(mu**2 - 1)) * np.sqrt((x_p - x_e)**2 - 1)
numerator = np.sqrt((-2 * x_p + 2 * x_e - 2)**2 - 4 * (mu**4 - 2 * mu**2 + 1) * (2 * x_e * x_p - x_p**2 - 2 * x_p - x_e**2 + 2 * x_e - 1)) + 2 * x_p - 2 * x_e + 2
denominator = 2 * (mu**4 - 2 * mu**2 + 1)
t_max_true = numerator / denominator
print(f'T Max True {t_max_true}')

# Generate parameter t
t_values = np.linspace(t_min, t_max_true, 1000, endpoint=True)
print(f'Last t {t_values[-1]}')

# Parametric equations
def compute_oval_points(t):
    x1 = x_p + ((mu**2 - 1) * t**2 - 2 * t - (x_p - x_e)**2 - 1) / (2 * (x_p - x_e))
    y1_pos = np.sqrt((t + 1)**2 - (x1 - x_p)**2)
    y1_neg = -np.sqrt((t + 1)**2 - (x1 - x_p)**2)
    x2 = -x_p - ((mu**2 - 1) * t**2 - 2 * t - (x_p - x_e)**2 - 1) / (2 * (x_p - x_e))
    y2_pos = np.sqrt((t + 1)**2 - (x2 + x_p)**2)
    y2_neg = -np.sqrt((t + 1)**2 - (x2 + x_p)**2)
    return (x1, y1_pos, y1_neg, x2, y2_pos, y2_neg)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-8, 8)
ax.set_ylim(-5, 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Cartesian Oval for Pursuer-Evader Problem')
ax.grid(True)

# Initialize the plot elements
scatter1_pos, = ax.plot([], [], 'm.', markersize=4)
scatter1_neg, = ax.plot([], [], 'm.', markersize=4)
scatter2_pos, = ax.plot([], [], 'g.', markersize=4)
scatter2_neg, = ax.plot([], [], 'g.', markersize=4)
pursuer1 = ax.plot(x_p, 0, 'bo', label='Pursuer 1')
pursuer2 = ax.plot(-x_p, 0, 'bo', label='Pursuer 2')
evader = ax.plot(x_e, 0, 'ro', label='Evader')
ax.legend()

# Initialize lists to store the trail of points
x1_trail, y1_pos_trail, y1_neg_trail = [], [], []
x2_trail, y2_pos_trail, y2_neg_trail = [], [], []

# Animation function
def animate(i):
    t = t_values[i]
    x1, y1_pos, y1_neg, x2, y2_pos, y2_neg = compute_oval_points(t)

    # Append current points to the trail lists
    x1_trail.append(x1)
    y1_pos_trail.append(y1_pos)
    y1_neg_trail.append(y1_neg)
    x2_trail.append(x2)
    y2_pos_trail.append(y2_pos)
    y2_neg_trail.append(y2_neg)

    # Update the scatter plot data
    scatter1_pos.set_data(x1_trail, y1_pos_trail)
    scatter1_neg.set_data(x1_trail, y1_neg_trail)
    scatter2_pos.set_data(x2_trail, y2_pos_trail)
    scatter2_neg.set_data(x2_trail, y2_neg_trail)
    return scatter1_pos, scatter1_neg, scatter2_pos, scatter2_neg

# Create animation
anim = FuncAnimation(fig, animate, frames=len(t_values), interval=10, blit=True)

# Save the animation as a GIF
anim.save('cartesian_oval_animation_with_trail.gif', writer=PillowWriter(fps=30))

plt.show()
