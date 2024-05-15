import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# Parameters
num_cutters = 2
dt = 0.1  # Time step
noise_level = 0.001  # Noise level
capture_radius = 0.5  # Radius within which the evader is considered captured
trail_length = 50  # Length of the ship trails (number of previous positions to show)

# Initial positions
cutter_positions = np.random.uniform(low=-10, high=10, size=(num_cutters, 2))  # Random initial positions of cutters
evader_position = np.random.uniform(low=0, high=1, size=(2,))  # Random initial position of evader

# Plot setup
plt.style.use('dark_background')
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)  # Initial x-axis limits
ax.set_ylim(-10, 10)  # Initial y-axis limits

# Initialize ship plots with empty lists for trails
cutters, = ax.plot([], [], 'bo', markersize=10, alpha=0.5)  # Cutter plots with fading trail
evader, = ax.plot([], [], 'ro', markersize=10, alpha=0.5)  # Evader plot with fading trail

# Draw capture radius circles around each cutter
capture_circles = [Circle((cutter_positions[i, 0], cutter_positions[i, 1]), capture_radius, linestyle='--', linewidth=1, edgecolor='blue', facecolor='none') for i in range(num_cutters)]
for circle in capture_circles:
    ax.add_patch(circle)

# Ship trail data
cutter_trails = [[] for _ in range(num_cutters)]
evader_trail = []

# Capture status for each cutter
captured_by_cutter = [False] * num_cutters

# Helper function to update positions and capture status
def update_ships(frame):
    global cutter_positions, evader_position, captured_by_cutter
        # Check if both cutters have captured the evader
    if any(captured_by_cutter):
        print("Evader captured by all cutters!")
        ani.event_source.stop()  # Stop the animation
    else:
        for i in range(num_cutters):
            if not captured_by_cutter[i]:  # Check if cutter hasn't captured evader
                if np.linalg.norm(cutter_positions[i] - evader_position) <= capture_radius:
                    print(f"Evader captured by cutter {i+1}!")
                    captured_by_cutter[i] = True  # Set capture status to True for this cutter
            
            
            midpoint = np.mean(cutter_positions, axis=0)
            cutter_vector = np.diff(cutter_positions, axis=0)[0]  # Vector from cutter 1 to cutter 2
            perpendicular_direction = np.array([-cutter_vector[1], cutter_vector[0]])  # Perpendicular to cutter vector
            # Update evader position along the perpendicular bisector
            evader_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)
            noise = noise_level * np.random.randn(2)
            evader_position += 0.2 * evader_direction * dt + noise
            # Update evader trail
            if len(evader_trail) == trail_length:
                evader_trail.pop(0)  # Remove oldest position
            evader_trail.append(tuple(evader_position))
    
        # Update cutter positions and trails
        for i in range(num_cutters):
            direction = evader_position - cutter_positions[i]
            noise = noise_level * np.random.randn(2)
            cutter_positions[i] += 0.6 * direction / np.linalg.norm(direction) * dt + noise
    
            # Update cutter trail
            if len(cutter_trails[i]) == trail_length:
                cutter_trails[i].pop(0)  # Remove oldest position
            cutter_trails[i].append(tuple(cutter_positions[i]))

        # Update plot data
        cutters.set_data(cutter_positions[:, 0], cutter_positions[:, 1])
        cutters.set_markersize(10 * capture_radius)
        
        evader.set_data(evader_position[0], evader_position[1])
        evader.set_markersize(10 * capture_radius)
        
        # Update plot limits to keep ships in view
        min_x = min(np.min(cutter_positions[:, 0]), evader_position[0]) - 1
        max_x = max(np.max(cutter_positions[:, 0]), evader_position[0]) + 1
        min_y = min(np.min(cutter_positions[:, 1]), evader_position[1]) - 1
        max_y = max(np.max(cutter_positions[:, 1]), evader_position[1]) + 1
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # Update capture radius circles
        for i in range(num_cutters):
            capture_circles[i].center = (cutter_positions[i, 0], cutter_positions[i, 1])
        
        for i in range(num_cutters):
            cutter_trail_x, cutter_trail_y = zip(*cutter_trails[i])
            ax.plot(cutter_trail_x, cutter_trail_y, 'b-', alpha=0.2)  # Cutter trail
        
        evader_trail_x, evader_trail_y = zip(*evader_trail)
        ax.plot(evader_trail_x, evader_trail_y, 'r-', alpha=0.2)  # Evader trail
        print(frame)
    return [cutters, evader] + capture_circles

# Create animation
ani = animation.FuncAnimation(fig, update_ships, frames=200, interval=50, blit=True)

plt.title('Two Cutters and Fugitive Ship Problem')
plt.xlabel('X')
plt.ylabel('Y')

# Save animation as GIF
print("Saving animation as a GIF...")
ani.save('cutters_and_fugitive.gif', writer='pillow', fps=20)

# Display animation
plt.show()
