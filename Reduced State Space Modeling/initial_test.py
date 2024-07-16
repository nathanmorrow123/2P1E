import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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
V_E = np.sqrt(2)     # Velocity of evader
chi = np.pi*3/4      # Angle chi P1 Heading
psi = np.pi/4        # Angle psi P2 heading
phi = 2*np.pi/6      # Angle phi Evader Heading
mu = V_E/V_P

initial_state = [0.5, 0, 0]  # Initial state [x_P, x_E, y_E] , y_E is calculated from sqrt(1-(x_P-x_E)^2)
initial_state[2] =  np.sqrt(1-(initial_state[0]-initial_state[1])**2)

t = np.linspace(0, 5, 10)  # Time vector

# Solve the differential equations
sol = odeint(dynamics, initial_state, t, args=(mu, chi, psi))

# Extract the results
x_P1 = sol[:, 0]
x_E = sol[:, 1]
y_E = sol[:, 2]
x_P2 = -1 * x_P1
y_P2 = np.zeros(len(t))
y_P1 = np.zeros(len(t))

# Plot the trajectories with a color gradient based on time
fig, ax = plt.subplots()
ax.scatter(x_P1, y_P1, label='Pursuer 1')
ax.scatter(x_P2, y_P2, label='Pursuer 2')
ax.scatter(x_E, y_E, label='Evader')

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Reduced State Space Trajectories')

# Show legend and grid
ax.legend()
ax.grid(True)
print(x_P1)
print(x_E)
plt.show()
