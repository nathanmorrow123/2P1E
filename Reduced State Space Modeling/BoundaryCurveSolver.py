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
x1 = np.linspace(0,x0,50)
y1 = np.linspace(0,z0,50)
nan_where = np.argwhere(np.isnan(z))
first_nan = nan_where[0]
print(x[first_nan-1])
print(z[first_nan-1])
x = x[0:int(first_nan-1)]
z = z[0:int(first_nan-1)]
# Perform polynomial regression to find the best-fit equation
degree = 2  # Degree of the polynomial
coeffs = np.polyfit(x, z, degree)
p = np.poly1d(coeffs)
print(coeffs)

# Plot the results
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(x, z, label='Numerical Solution (RK4)')
#plt.plot(x, p(x), label=f'Best Fit Polynomial (degree {degree})')
plt.plot(x1,y1)
plt.xlabel('$x_P$')
plt.xticks([0,1/np.sqrt(2),1, x[-1]])
plt.yticks([0,1/np.sqrt(2),1])
plt.gca().set_aspect('equal')
plt.ylabel('$z$')
plt.title('Numerical Solution of the Boundary Curve')
plt.legend()
plt.grid(True)
plt.savefig('Boundary_Curve.pdf')
plt.show()

