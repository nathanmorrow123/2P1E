import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Parameters
num_cutters = 8
dt = 0.1  # Time step
noise_level = 0.0001  # Noise level
capture_radius = 0.5  # Radius within which the fugitive is considered captured
trail_length = 10  # Length of the ship trails (number of previous positions to show)
v_E = 1.2
v_P = 1

# Initial positions
cutter_positions = 10*np.random.vonmises((0,0),0.1,size=(num_cutters, 2))  # Random initial positions of cutters

fugitive_position = np.random.uniform(low=-1, high=-1, size=(2,))  # Random initial position of fugitive

# Plot setup
plt.style.use('dark_background')
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)  # Initial x-axis limits
ax.set_ylim(0, 20)  # Initial y-axis limits

# Initialize ship plots with empty lists for trails
cutters, = ax.plot([], [], 'bo', markersize=10, alpha=0.5)  # Cutter plots with fading trail
fugitive, = ax.plot([], [], 'ro', markersize=10, alpha=0.5)  # Evader plot with fading trail

# Draw capture radius circles around each cutter
capture_circles = [Circle((cutter_positions[i, 0], cutter_positions[i, 1]), capture_radius, linestyle='--', linewidth=1, edgecolor='darkblue', facecolor='none') for i in range(num_cutters)]
for circle in capture_circles:
    ax.add_patch(circle)

# Ship trail data
cutter_trails = [[] for _ in range(num_cutters)]
fugitive_trail = []

# Capture status for each cutter
captured_by_cutter = [False] * num_cutters

def update_fugitive_position():
    global fugitive_position

    # Calculate all pairwise distances between cutters
    distances = np.linalg.norm(cutter_positions[:, np.newaxis] - cutter_positions, axis=2)
    np.fill_diagonal(distances, 0)  # Ignore self-distances

    # Find the two cutters that are farthest apart
    cutter1_idx, cutter2_idx = np.unravel_index(np.argmax(distances), distances.shape)
    cutter1_pos, cutter2_pos = cutter_positions[cutter1_idx], cutter_positions[cutter2_idx]
    # Calculate the midpoint and perpendicular direction
    midpoint = (cutter1_pos + cutter2_pos) / 2
    cutter_vector = cutter2_pos - cutter1_pos
    perpendicular_direction = np.array([-cutter_vector[1], cutter_vector[0]])

    # Update fugitive position along the perpendicular bisector
    fugitive_direction = (fugitive_position - midpoint) / np.linalg.norm(fugitive_position - midpoint)
    # fugitive_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)
    noise = noise_level * np.random.randn(2)
    fugitive_position += v_E * fugitive_direction * dt + noise
    # Update fugitive trail
    if len(fugitive_trail) == trail_length:
        fugitive_trail.pop(0)  # Remove oldest position
    fugitive_trail.append(tuple(fugitive_position))

def update_cutter_positions():
    for i in range(num_cutters):
        if not captured_by_cutter[i]:
            direction = fugitive_position - cutter_positions[i]
            noise = noise_level * np.random.randn(2)
            cutter_positions[i] += v_P * direction / np.linalg.norm(direction) * dt + noise
            # Update cutter trail
            if len(cutter_trails[i]) == trail_length:
                cutter_trails[i].pop(0)  # Remove oldest position
            cutter_trails[i].append(tuple(cutter_positions[i]))

def update_capture_status():
    global capture_buffer_counter

    for i in range(num_cutters):
        if not captured_by_cutter[i]:  # Check if cutter hasn't captured fugitive
            if np.linalg.norm(cutter_positions[i] - fugitive_position) <= capture_radius:
                print(f"Evader captured by cutter {i+1}!")
                captured_by_cutter[i] = True  # Set capture status to True for this cutter

    if all(captured_by_cutter):
        if capture_buffer_counter < 10:
            capture_buffer_counter += 1
            print("Evader captured by all cutter!")
        else:
            ani.event_source.stop()  # Stop the animation

def update_plot_data():
    cutters.set_data(cutter_positions[:, 0], cutter_positions[:, 1])
    cutters.set_markersize(10 * capture_radius)

    fugitive.set_data(fugitive_position[0], fugitive_position[1])
    fugitive.set_markersize(10 * capture_radius)

    # Calculate new limits to maintain equal aspect ratio
    all_x = np.concatenate((cutter_positions[:, 0], [fugitive_position[0]]))
    all_y = np.concatenate((cutter_positions[:, 1], [fugitive_position[1]]))
    x_min, x_max = np.min(all_x) - 1, np.max(all_x) + 1
    y_min, y_max = np.min(all_y) - 1, np.max(all_y) + 1

    # Find the range and set limits to keep the aspect ratio equal
    data_range = max(x_max - x_min, y_max - y_min)
    ax.set_xlim((x_min, x_min + data_range))
    ax.set_ylim((y_min, y_min + data_range))

    # Update capture radius circles
    for i in range(num_cutters):
        capture_circles[i].center = (cutter_positions[i, 0], cutter_positions[i, 1])

    # Plot trails
    for i in range(num_cutters):
        cutter_trail_x, cutter_trail_y = zip(*cutter_trails[i])
        ax.plot(cutter_trail_x, cutter_trail_y, 'b-', alpha=0.2)  # Cutter trail

    if fugitive_trail:
        fugitive_trail_x, fugitive_trail_y = zip(*fugitive_trail)
        ax.plot(fugitive_trail_x, fugitive_trail_y, 'r-', alpha=0.2)  # Evader trail

def update_ships(frame):
    if not any(captured_by_cutter):
        update_fugitive_position()
    update_cutter_positions()
    update_capture_status()
    update_plot_data()
    print(frame)
    return [cutters, fugitive] + capture_circles

# Initialize the capture buffer counter
capture_buffer_counter = 0

# Create animation
ani = animation.FuncAnimation(fig, update_ships, frames=250, blit=True)

plt.title('Cutters and Evader Simulation')
plt.xlabel('$X$')
plt.ylabel('$Y$')

# Save animation as GIF
print("Saving animation as a GIF...")
ani.save('cutters_and_fugitive.gif', writer='pillow', fps=30)

# Display animation
plt.show()
