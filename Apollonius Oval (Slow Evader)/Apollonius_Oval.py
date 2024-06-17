import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_interception_point(evader, pursuer, mu, capture_radius, chi):
    """
    Calculate interception point for given evader and pursuer, speed ratio, and capture radius.

    evader, pursuer: Tuples representing the coordinates of the evader (x1, y1) and pursuer (x2, y2)
    mu: The speed ratio of the evader to the pursuer
    capture_radius: The capture radius for the pursuer
    chi: The heading angle of the pursuer
    """
    x1, y1 = evader
    x2, y2 = pursuer

    # Calculate evader and pursuer directions
    alpha = np.arccos(1 / mu)
    phi = chi - alpha
    evader_direction = np.array([np.cos(phi), np.sin(phi)])
    pursuer_direction = np.array([np.cos(chi), np.sin(chi)])

    # Calculate relative velocities
    evader_velocity = mu * evader_direction
    pursuer_velocity = pursuer_direction

    # Solve for time to interception considering the capture radius
    relative_velocity = evader_velocity - pursuer_velocity

    # Calculate the distance between the evader and the pursuer
    initial_distance = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
    interception_time = np.roots([np.linalg.norm(evader_velocity)**2-np.linalg.norm(pursuer_velocity)**2,
                                    2*(np.linalg.norm(evader)-np.linalg.norm(pursuer))*np.linalg.norm(evader_velocity),
                                        (np.linalg.norm(evader)-np.linalg.norm(pursuer))**2-capture_radius**2])
    l = capture_radius
    d = initial_distance
    mu = 1/mu
    # Calculate interception time based on relative velocities and capture radius
    if(np.cos(phi)<np.sqrt(1-mu**2)*np.sqrt(1-(l/d)**2)-mu*(l/d) and np.cos(-phi)<np.sqrt(1-mu**2)*np.sqrt(1-(l/d)**2)-mu*(l/d)):
        interception_point = np.array([x1, y1]) + interception_time[0] * evader_velocity
    else:
        interception_point = np.array([x1, y1]) + interception_time[1] * evader_velocity
    return interception_point

# Example usage
evader = np.array([0, 0])
pursuer = np.array([2, 0])
mu = 2  # Evader is twice as fast as the pursuer
capture_radius = 1
num_frames = 100

# Create a blank figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid(True)

# Initialize the interception point
interception_point, = ax.plot([], [], 'g.', label='Interception Point')

# Plot the positions of the evader and the pursuer
ax.plot(*evader, 'ro', label='Evader')
ax.plot(*pursuer, 'bo', label='Pursuer')

# Draw capture radius circles around the pursuer
circle = plt.Circle(pursuer, capture_radius, color='b', alpha=0.3)
ax.add_artist(circle)

# Initialize evader and pursuer trajectories as quiver plots
evader_path = ax.quiver(evader[0], evader[1], 0, 0, color='r', scale=1, scale_units='xy', angles='xy', label='Evader Trajectory')
pursuer_path = ax.quiver(pursuer[0], pursuer[1], 0, 0, color='b', scale=1, scale_units='xy', angles='xy', label='Pursuer Trajectory')

def update(frame):
    # Calculate interception point for the current heading angle
    theta = frame * 2 * np.pi / num_frames
    point = calculate_interception_point(evader, pursuer, mu, capture_radius, theta)
    interception_point.set_data(point)

    # Calculate evader and pursuer trajectories
    evader_path.set_UVC(point[0] - evader[0], point[1] - evader[1])
    pursuer_path.set_UVC(point[0] - pursuer[0], point[1] - pursuer[1])

    return interception_point, evader_path, pursuer_path

# Animate the interception point
ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interception Point Animation for Evader and Pursuer Problem with Capture Radius')

# Save the animation as a video file
ani.save('interception_animation_quiver.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()
