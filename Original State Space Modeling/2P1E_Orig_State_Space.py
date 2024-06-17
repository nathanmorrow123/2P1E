import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Parameters
num_pursurers = 2
dt = 0.1  # Time step
noise_level = 0.0001  # Noise level
capture_radius = 0.5  # Radius within which the evader is considered captured
trail_length = 10  # Length of the ship trails (number of previous positions to show)
v_E = np.sqrt(2)
v_P = 1

# Initial positions
# pursurer_positions = np.random.uniform(low=-1, high=1, size=(num_pursurers, 2))  # Random initial positions of pursurers
# evader_position = np.random.uniform(low=-5, high=-5, size=(2,))  # Random initial position of evader

pursurer_positions = np.array([[2.0,10.0],[-2.0,10.0]])
evader_position = np.array([0.0,0.0])

# Plot setup
plt.style.use('dark_background')
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)  # Initial x-axis limits
ax.set_ylim(0, 20)  # Initial y-axis limits

# Initialize ship plots with empty lists for trails
pursurers, = ax.plot([], [], 'bo', markersize=10, alpha=0.5)  # Cutter plots with fading trail
evader, = ax.plot([], [], 'ro', markersize=10, alpha=0.5)  # Evader plot with fading trail

# Draw capture radius circles around each pursurer
capture_circles = [Circle((pursurer_positions[i, 0], pursurer_positions[i, 1]), capture_radius, linestyle='--', linewidth=1, edgecolor='darkblue', facecolor='none') for i in range(num_pursurers)]
for circle in capture_circles:
    ax.add_patch(circle)

# Ship trail data
pursurer_trails = [[] for _ in range(num_pursurers)]
evader_trail = []

# Capture status for each pursurer
captured_by_pursurer = [False] * num_pursurers

# Helper function to update positions and capture status
def update(frame):
    global pursurer_positions, evader_position, captured_by_pursurer, capture_buffer_counter

    if any(captured_by_pursurer):
        if capture_buffer_counter < 10:
            capture_buffer_counter += 1
            print("Evader captured by a pursurer!")
        else:
            ani.event_source.stop()  # Stop the animation
    else:
        for i in range(num_pursurers):
            if not captured_by_pursurer[i]:  # Check if pursurer hasn't captured evader
                if np.linalg.norm(pursurer_positions[i] - evader_position) <= capture_radius:
                    print(f"Evader captured by pursurer {i+1}!")
                    captured_by_pursurer[i] = True  # Set capture status to True for this pursurer

        if not all(captured_by_pursurer):
            midpoint = np.mean(pursurer_positions, axis=0)
            pursurer_vector = np.diff(pursurer_positions, axis=0)[0]  # Vector from pursurer 1 to pursurer 2
            perpendicular_direction = np.array([-pursurer_vector[1], -pursurer_vector[0]])  # Perpendicular to pursurer vector
            # Update evader position along the perpendicular bisector
            evader_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)
            noise = noise_level * np.random.randn(2)
            evader_position += v_E * evader_direction * dt + noise
            # Update evader trail
            if len(evader_trail) == trail_length:
                evader_trail.pop(0)  # Remove oldest position
            evader_trail.append(tuple(evader_position))

        # Update pursurer positions and trails
        for i in range(num_pursurers):
            direction = evader_position - pursurer_positions[i]
            noise = noise_level * np.random.randn(2)
            pursurer_positions[i] += v_P * direction / np.linalg.norm(direction) * dt + noise

            # Update pursurer trail
            if len(pursurer_trails[i]) == trail_length:
                pursurer_trails[i].pop(0)  # Remove oldest position
            pursurer_trails[i].append(tuple(pursurer_positions[i]))

        # Update plot data
        pursurers.set_data(pursurer_positions[:, 0], pursurer_positions[:, 1])
        pursurers.set_markersize(10 * capture_radius)

        evader.set_data(evader_position[0], evader_position[1])
        evader.set_markersize(10 * capture_radius)

        # Calculate new limits to maintain equal aspect ratio
        all_x = np.concatenate((pursurer_positions[:, 0], [evader_position[0]]))
        all_y = np.concatenate((pursurer_positions[:, 1], [evader_position[1]]))
        x_min, x_max = np.min(all_x) - 1, np.max(all_x) + 1
        y_min, y_max = np.min(all_y) - 1, np.max(all_y) + 1

        # Find the range and set limits to keep the aspect ratio equal
        data_range = max(x_max - x_min, y_max - y_min)
        ax.set_xlim((x_min, x_min + data_range))
        ax.set_ylim((y_min, y_min + data_range))
        
        # Update capture radius circles
        for i in range(num_pursurers):
            capture_circles[i].center = (pursurer_positions[i, 0], pursurer_positions[i, 1])

        for i in range(num_pursurers):
            pursurer_trail_x, pursurer_trail_y = zip(*pursurer_trails[i])
            ax.plot(pursurer_trail_x, pursurer_trail_y, 'b-', alpha=0.2)  # Cutter trail

        evader_trail_x, evader_trail_y = zip(*evader_trail)
        ax.plot(evader_trail_x, evader_trail_y, 'r-', alpha=0.2)  # Evader trail
        print(frame)
    return [pursurers, evader] + capture_circles

# Initialize the capture buffer counter
capture_buffer_counter = 0

# Create animation
ani = animation.FuncAnimation(fig, update, frames=250, blit=True)

plt.title('Two Cutters and Fugitive Ship Problem')
plt.xlabel('$X$')
plt.ylabel('$Y$')

# Save animation as GIF
print("Saving animation as a GIF...")
ani.save('pursurers_and_fugitive.gif', writer='pillow', fps=30)

# Display animation
plt.show()