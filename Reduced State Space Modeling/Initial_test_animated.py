import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from matplotlib.patches import Circle

# Define the dynamics function for the system
def dynamics(state, t, mu, chi, psi):
    x_P, x_E, y_E = state
    
    # Unpack angles
    cos_chi = np.cos(chi) # (Original)
    # cos_chi = -(x_P-x_E) # Enforcing PP
    sin_chi = np.sin(chi) # (Original)
    # sin_chi = np.sqrt(1-(x_P-x_E)**2) # Enforcing PP
    cos_psi = np.cos(psi)  # (Original)
    sin_psi = np.sin(psi)  # (Original)

    # Define the dynamics equations
    dx_P = 0.5 * (cos_chi - cos_psi)
    dx_E = mu * np.cos(phi) - 0.5 * (cos_chi + cos_psi) + 0.5 * (y_E / x_P) * (sin_chi - sin_psi)
    dy_E = mu * np.sin(phi) - 0.5 * (sin_chi + sin_psi) - 0.5 * (x_E / x_P) * (sin_chi - sin_psi)

    return [dx_P, dx_E, dy_E]

# Set initial conditions and parameters
V_P = 1.0            # Velocity of pursuers
V_E = np.sqrt(2)           # Velocity of evader
chi = np.pi*3/4      # Angle chi P1 Heading
psi = np.pi/4       # Angle psi P2 heading
phi = 2*np.pi/6     # Angle phi Evader Heading
mu = V_E/V_P

initial_state = [2, 0, 1]  # Initial state [x_P, x_E, y_E]
t = np.linspace(0, 10, 100)  # Time vector

# Solve the differential equations
sol = odeint(dynamics, initial_state, t, args=(mu, chi, psi))

# Extract the results
x_P1 = sol[:, 0]
x_E = sol[:, 1]
y_E = sol[:, 2]
x_P2 = -1 * x_P1
y_P2 = np.zeros(len(t))
y_P1 = np.zeros(len(t))

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('$x_E$')
ax.set_ylabel('$y_E$')
ax.set_title('Pursuer-Evader Dynamics in Reduced State Space')

# Plot the tails
tail_length = 5
tail_x_P1 = []
tail_y_P1 = []
tail_x_P2 = []
tail_y_P2 = []
tail_x_E = []
tail_y_E = []

tail_P1, = ax.plot(tail_x_P1, tail_y_P1, 'bo', alpha=0.2)
tail_P2, = ax.plot(tail_x_P2, tail_y_P2, 'bo', alpha=0.2)
tail_E, = ax.plot(tail_x_E, tail_y_E,'ro', alpha=0.2)

# Plot the capture circles
capture_radius = 1
P1_CC = Circle((x_P1[0], y_P1[0]), capture_radius, linestyle='--', linewidth=1, edgecolor='darkblue', facecolor='none')
P2_CC = Circle((x_P2[0], y_P2[0]), capture_radius, linestyle='--', linewidth=1, edgecolor='darkblue', facecolor='none')
ax.add_patch(P1_CC)
ax.add_patch(P2_CC)

# Plot the current positions
P1, = ax.plot(x_P1[0], y_P1[0], 'bo',label='Pursuer 1')
P2, = ax.plot(x_P2[0], y_P2[0], 'bo',label='Pursuer 2')
E, = ax.plot(x_E[0], y_E[0], 'ro',label='Evader')

ax.legend()

# Update function for animation
def update(frame):
    P1.set_data(x_P1[frame], y_P1[frame])
    P2.set_data(x_P2[frame], y_P2[frame])
    E.set_data(x_E[frame], y_E[frame])
    if len(tail_x_P1)>tail_length:
        tail_x_P1.pop(0)
        tail_y_P1.pop(0)
        tail_x_P2.pop(0)
        tail_y_P2.pop(0)
        tail_x_E.pop(0)
        tail_y_E.pop(0)
    tail_x_P1.append(x_P1[frame])
    tail_y_P1.append(y_P1[frame])
    tail_x_P2.append(x_P2[frame])
    tail_y_P2.append(y_P2[frame])
    tail_x_E.append(x_E[frame])
    tail_y_E.append(y_E[frame])
    tail_P1.set_data(tail_x_P1, tail_y_P1)
    tail_P2.set_data(tail_x_P2, tail_y_P2)
    tail_E.set_data(tail_x_E, tail_y_E)

    # Update capture circles
    P1_CC.center = (x_P1[frame], y_P1[frame])
    P2_CC.center = (x_P2[frame], y_P2[frame])

    # Calculate new limits to maintain equal aspect ratio
    all_x = np.concatenate((x_P1[:frame+1], x_P2[:frame+1], x_E[:frame+1]))
    all_y = np.concatenate((y_P1[:frame+1], y_P2[:frame+1], y_E[:frame+1]))
    x_min, x_max = np.min(all_x) - 1, np.max(all_x) + 1
    y_min, y_max = np.min(all_y) - 1, np.max(all_y) + 1

    # Find the range and set limits to keep the aspect ratio equal
    data_range = max(x_max - x_min, y_max - y_min)
    ax.set_xlim((x_min, x_min + data_range))
    ax.set_ylim((y_min, y_min + data_range))

    return P1, P2, E, tail_P1, tail_P2, tail_E, P1_CC, P2_CC

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)
ani.save('ReducedStateSpace.gif')
plt.show()
