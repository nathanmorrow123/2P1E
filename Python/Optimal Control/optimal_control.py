import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = np.sqrt(2)  # (V_e / V_p)
T = 5  # Total time
dt = 0.01  # Time step
n_steps = int(T / np.abs(dt))
n_lines = 7

# Initial conditions for reduced state space variables
x_p0 = np.linspace(np.sqrt((mu**2 - 1) / mu), 1, n_lines, endpoint=False)
z_0 = x_p0

# Terminal conditions for costates
lambda_x_p_0 = -1 / (x_p0 - np.sqrt(mu**2 - 1) * np.sqrt(1 - x_p0))
lambda_z_0 = 1 / (x_p0 - np.sqrt(mu**2 - 1) * np.sqrt(1 - x_p0))

# Arrays to store results
x_p_arr = np.zeros((n_steps + 1, len(x_p0)))
z_arr = np.zeros((n_steps + 1, len(z_0)))
x_p_arr[0] = x_p0
z_arr[0] = z_0

# Forward integration of state equations
for i in range(len(x_p0)):
    # Assign initial variables
    x_p = x_p0[i]
    z = z_0[i]
    lambda_x_p = lambda_x_p_0[i]
    lambda_z = lambda_z_0[i]
    
    for t in range(n_steps):
        # State equations
        x_p_dot = (1 / 2) * (z - lambda_x_p / np.sqrt(lambda_x_p**2 + (1 - z**2) / x_p * lambda_z**2))
        z_dot = (mu**2 - 1) * np.sqrt(1 - z**2) + (1 / 2) * ((1 - z**2) / x_p**2) * (x_p + lambda_z / np.sqrt(lambda_z**2 * (1 - z**2) / x_p**2 + lambda_x_p**2))

        x_p += x_p_dot * dt
        z += z_dot * dt

        x_p_arr[t + 1, i] = x_p
        z_arr[t + 1, i] = z

        # Reverse integration of costate equations
        lambda_x_p_dot = (1 / 2) * lambda_z * (1 - z**2) / x_p**3 * (lambda_z / np.sqrt(lambda_x_p**2 + (1 - z**2) / x_p**2 * lambda_z**2) - x_p)
        lambda_z_dot = (1 / 2) * ((lambda_z**2 / x_p**2 * z) / np.sqrt(lambda_x_p**2 + (1 - z**2) / x_p**2 * lambda_z**2)) + (1 / 2) * lambda_x_p - z * lambda_z / x_p - np.sqrt(mu**2 - 1) * z / (1 - z**2) * lambda_z

        lambda_x_p += lambda_x_p_dot * dt
        lambda_z += lambda_z_dot * dt

x_p_arr = np.array(x_p_arr)
z_arr = np.array(z_arr)
#Cleaning Arrays
x_p_arr = x_p_arr[~np.isnan(x_p_arr)]
z_arr = z_arr[~np.isnan(z_arr)]

# Finite boundary lines
x0 = np.sqrt(mu**2 - 1) / mu  # Initial x value
z0 = np.sqrt(mu**2 - 1) / mu  # Initial z value
x1 = np.linspace(0, x0, 50)
z1 = np.linspace(0, z0, 50)
x2 = np.linspace(x0, 1, 50)
z2 = np.linspace(z0, 1, 50)
x3 = np.linspace(1, 1.454, 50)
z3 = np.ones(50)

def add_arrow(line, position=None, direction='right', size=35, color=None):
    """
    Add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    line = line[0]
    color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    print(color,xdata,ydata)
    if position is None:
        position = ydata.mean()
    
    # find closest index
    start_ind = int(len(xdata) / 2)
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    
    line.axes.annotate(
        '',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

# Plotting the results
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('dark_background')
plt.set_cmap('cool')
for i in range(n_lines):
    line = plt.plot(x_p_arr[:, i], z_arr[:, i])
    add_arrow(line, direction='left')
plt.plot(x1, z1, color='white')
plt.plot(x2, z2, color='white')
plt.plot(x3, z3, color='white')
plt.xticks([0, 1 / np.sqrt(2), 1, 1.454])
plt.yticks([0, 1 / np.sqrt(2), 1])
plt.axis('equal')
plt.xlabel('$x_p$')
plt.ylabel('$z$')
plt.title('Optimal Control Flow Field')
plt.grid(alpha=0.3)
plt.savefig('Optimal_Control_Flow_Field.pdf')
plt.show()
