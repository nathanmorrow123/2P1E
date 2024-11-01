import numpy as np
import matplotlib.pyplot as plt

# Parameters
v1 = 1.0  # Speed of the pursuer
v2 = np.sqrt(2)  # Speed of the evader
T = 10.0  # Total time
dt = 0.01 # Time step
n_steps = int(T / dt)

# Initial and terminal conditions
x1, y1 = 0.0, 1.0  # Initial position of the pursuer
x2, y2 = 0.0, 0.0  # Initial position of the evader

# Terminal conditions for costates
lambda1, lambda2 = 0.0, 0.0

# Arrays to store results
x1_arr, y1_arr = [x1], [y1]
x2_arr, y2_arr = [x2], [y2]
lambda1_arr, lambda2_arr = [lambda1], [lambda2]
theta_arr = []

# Function to compute optimal heading angle
def optimal_control(lambda1, lambda2):
    return np.arctan2(lambda2, lambda1)

# Evader's flight path: sinusoidal variation of heading angle
def evader_heading(t):
    return np.pi / 4 * np.sin(0.5 * t) + np.random.randn()

# Forward integration of state equations
for t in range(n_steps):
    theta = optimal_control(lambda1, lambda2)
    
    # State equations
    x1_dot = v1 * np.cos(theta)
    y1_dot = v1 * np.sin(theta)
    phi = evader_heading(t * dt)  # Update evader's heading angle
    x2_dot = v2 * np.cos(phi)
    y2_dot = v2 * np.sin(phi)
    
    x1 += x1_dot * dt
    y1 += y1_dot * dt
    x2 += x2_dot * dt
    y2 += y2_dot * dt
    
    x1_arr.append(x1)
    y1_arr.append(y1)
    x2_arr.append(x2)
    y2_arr.append(y2)
    theta_arr.append(theta)
    
    # Reverse integration of costate equations
    lambda1 += (-2 * (x1 - x2)) * dt
    lambda2 += (-2 * (y1 - y2)) * dt
    
    lambda1_arr.append(lambda1)
    lambda2_arr.append(lambda2)

# Convert theta_arr to numpy array for vectorized operations
theta_arr = np.array(theta_arr)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x1_arr, y1_arr, label='Pursuer')
plt.plot(x2_arr, y2_arr, label='Evader')
plt.quiver(x1_arr[:-1], y1_arr[:-1], np.cos(theta_arr), np.sin(theta_arr), scale=30, width=0.003, color='r', alpha=0.5, label='Pursuer Heading')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pursuit Problem with Complex Evader Path')
plt.grid()
plt.show()
