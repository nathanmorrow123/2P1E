import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d

# Parameters
num_cutters = 2
dt = 0.1  # Time step
noise_level = 0.001  # Noise level
capture_radius = 0.5  # Radius within which the evader is considered captured
trail_length = 50  # Length of the ship trails (number of previous positions to show)

# Initial positions
cutter_positions = np.random.uniform(low=-10, high=10, size=(num_cutters, 3))  # Random initial positions of cutters in 3D
evader_position = np.random.uniform(low=-10, high=10, size=(3,))  # Random initial position of evader in 3D

# Plot setup
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D axes

# Initialize ship plots with empty lists for trails
cutters_plot, = ax.plot([], [], [], 'bo', markersize=10, alpha=0.5)  # Cutter plots with fading trail
evader_plot, = ax.plot([], [], [], 'ro', markersize=10, alpha=0.5)  # Evader plot with fading trail
capture_sphers_plot1 = ax.plot_wireframe([],[],[],'b')
capture_sphers_plot2 = ax.plot_wireframe([],[],[],'b')

# Ship trail data
cutter_trails = [[] for _ in range(num_cutters)]
evader_trail = []

# Capture status for each cutter
captured_by_cutter = [False] * num_cutters

# Helper function to update positions and capture status
def update_ships(frame):
    global cutter_positions, evader_position, captured_by_cutter
    
    # Check if all cutters have captured the evader
    if all(captured_by_cutter):
        print("Evader captured by all cutters!")
        ani.event_source.stop()  # Stop the animation
    else:
        for i in range(num_cutters):
            if not captured_by_cutter[i]:  # Check if cutter hasn't captured evader
                if np.linalg.norm(cutter_positions[i] - evader_position) <= capture_radius:
                    print(f"Evader captured by cutter {i+1}!")
                    captured_by_cutter[i] = True  # Set capture status to True for this cutter
            
            # Update evader position based on your logic
            midpoint = np.mean(cutter_positions, axis=0)
            cutter_vector = np.diff(cutter_positions, axis=0)[0]  # Vector from cutter 1 to cutter 2
            perpendicular_direction = np.cross(cutter_vector, np.array([0, 0, 1]))  # Perpendicular to cutter vector
            evader_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)
            noise = noise_level * np.random.randn(3)
            evader_position += 0.2 * evader_direction * dt + noise
            evader_trail.append(tuple(evader_position))
    
        # Update cutter positions and trails
        for i in range(num_cutters):
            direction = evader_position - cutter_positions[i]
            noise = noise_level * np.random.randn(3)
            cutter_positions[i] += 0.6 * direction / np.linalg.norm(direction) * dt + noise
            cutter_trails[i].append(tuple(cutter_positions[i]))

        # Ensure trail length is within limit
        if len(evader_trail) > trail_length:
            evader_trail.pop(0)
        for trail in cutter_trails:
            if len(trail) > trail_length:
                trail.pop(0)

        # Update plot data
        # Update cutters plot
        cutter_positions_array = np.array(cutter_positions)
        cutters_plot.set_data(cutter_positions_array[:, 0], cutter_positions_array[:, 1])
        cutters_plot.set_3d_properties(cutter_positions_array[:, 2])

        # Update evader plot
        tempArray = np.array([evader_position,evader_position])
        evader_plot.set_data(tempArray[:,0], tempArray[:,1])
        evader_plot.set_3d_properties(tempArray[:,2])
        
        # Update plot limits to keep ships in view
        min_x = min(np.min(cutter_positions[:, 0]), evader_position[0]) - 1
        max_x = max(np.max(cutter_positions[:, 0]), evader_position[0]) + 1
        min_y = min(np.min(cutter_positions[:, 1]), evader_position[1]) - 1
        max_y = max(np.max(cutter_positions[:, 1]), evader_position[1]) + 1
        min_z = min(np.min(cutter_positions[:, 2]), evader_position[2]) - 1
        max_z = max(np.max(cutter_positions[:, 2]), evader_position[2]) + 1
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z,max_z)
        
        # Plot trails
        for i in range(num_cutters):
            if len(cutter_trails[i]) > 1:
                cutter_trail_x, cutter_trail_y, cutter_trail_z = zip(*cutter_trails[i])
                ax.plot(cutter_trail_x, cutter_trail_y, cutter_trail_z, 'b-', alpha=0.2)  # Cutter trail
        
        if len(evader_trail) > 1:
            evader_trail_x, evader_trail_y, evader_trail_z = zip(*evader_trail)
            ax.plot(evader_trail_x, evader_trail_y, evader_trail_z, 'r-', alpha=0.2)  # Evader trail
        
        # Plot capture spheres
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = capture_radius*np.cos(u)*np.sin(v) + cutter_positions[:,0]
        y = capture_radius*np.sin(u)*np.sin(v) + cutter_positions[:,1]
        z = capture_radius*np.cos(v) + cutter_positions[:,2]
        capture_sphers_plot1.set_data(x[0],y[0])
        capture_sphers_plot2.set_3d_properties(z[0])
        capture_sphers_plot1.set_data(x[1],y[1])
        capture_sphers_plot2.set_3d_properties(z[1])
        
        print(frame)  # Print frame number for debugging
    return cutters_plot, evader_plot, capture_sphers_plot1, capture_sphers_plot2

# Create animation
ani = animation.FuncAnimation(fig, update_ships, frames=200, interval=50, blit=True)

ax.set_title('Two Cutters and Fugitive Ship Problem')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Save animation as GIF
print("Saving animation as a GIF...")
ani.save('cutters_and_fugitive_3d.gif', writer='pillow', fps=20)

# Display animation
plt.show()
