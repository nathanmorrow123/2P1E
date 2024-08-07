import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation as a function
def dz_dx(x, z):
    return (2 * np.sqrt(1 + np.sqrt(1 - z**2) / x) - z * (2 + np.sqrt(1 - z**2) / x)) / np.sqrt(1 - z**2)

# Runge-Kutta 4th order method implementation
def runge_kutta_4(f, x0, z0, x_end, h):
    n = int((x_end - x0) / h)
    x = np.linspace(x0, x_end, n+1)
    z = np.zeros(n+1)
    z[0] = z0
    
    for i in range(n):
        k1 = h * f(x[i], z[i])
        k2 = h * f(x[i] + h/2, z[i] + k1/2)
        k3 = h * f(x[i] + h/2, z[i] + k2/2)
        k4 = h * f(x[i] + h, z[i] + k3)
        
        z[i+1] = z[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x, z

# Initial conditions and parameters
x0 = 1/np.sqrt(2)  # Initial x value
z0 = 1/np.sqrt(2)  # Initial z value
x_end = 1.5  # Ending x value
h = 0.00001  # Step size


# Perform the numerical integration using RK4
x, z = runge_kutta_4(dz_dx, x0, z0, x_end, h)

#Finding the last x position in solution
nan_where = np.argwhere(np.isnan(z))
first_nan = nan_where[0]
print(x[first_nan-1])
print(z[first_nan-1])
x = x[0:int(first_nan-1)]
z = z[0:int(first_nan-1)]

x_elip = [[],[],[]]
z_elip = [[],[],[]]
i = 0
x0_arr = [x0-0.1,x0,x0+0.1]
# Finding the ellipse lines
for x0 in x0_arr:
    for theta in np.linspace(np.pi/2,np.pi,1000):
        xe = (x0+np.sqrt(1-z0)) + np.cos(theta)
        ze = np.sin(theta)
        if ((xe <= np.sqrt(2) and ze <= 1 ) and (np.abs(ze/xe) <= 1) and (xe >= 1/np.sqrt(2))):
            x_elip[i].append(xe)
            z_elip[i].append(ze)
        
    i += 1

x0 = 1/np.sqrt(2)  # Initial x value
z0 = 1/np.sqrt(2)  # Initial z value
print(x_elip)
print(z_elip)     


# Finite Boundry lines
x1 = np.linspace(0,x0,50)
z1 = np.linspace(0,z0,50)
x2 = np.linspace(x0,1,50)
z2 = np.linspace(z0,1,50)
x3 = np.linspace(1,x[-1],50)
z3 = np.ones(50)

# Perform polynomial regression to find the best-fit equation
degree = 3  # Degree of the polynomial
coeffs = np.polyfit(x, z, degree)
p = np.poly1d(coeffs)
print(coeffs)

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

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

    if position is None:
        position = ydata.mean()
    # find closest index
    start_ind = int(len(xdata)/2)
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


# Plot the results
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "Times New Roman"
plt.set_cmap('cool')
line = plt.plot(x, z, label='Numerical Solution (RK4)')
add_arrow(line, direction='left')

for i in [0,1,2]:
    line = plt.plot(x_elip[i],z_elip[i], label = f'Circular arc with $x_0$  = {np.round(x0_arr[i],4)}')
    add_arrow(line)
plt.plot(x1,z1, color = 'black')
plt.plot(x2,z2, color = 'black')
plt.plot(x3,z3, color = 'black')

plt.xlabel('$x_P$')
plt.xticks([0,1/np.sqrt(2),1, x[-1]])
plt.yticks([0,1/np.sqrt(2),1])
plt.gca().set_aspect('equal')
plt.ylabel('$z$')
plt.title('Sub-Optimal Flow Field, $\mu = \sqrt{2}$')
plt.legend()
plt.grid(True)
plt.savefig('Barrier_Curve.pdf')
plt.show()

