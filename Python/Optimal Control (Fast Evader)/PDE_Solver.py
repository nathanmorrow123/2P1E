import numpy as np
import matplotlib.pyplot as plt
from pde import PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid
from sympy import symbols, sqrt

# Define the symbols
lambda_xp, lambda_z, x_P, z, mu = symbols('lambda_xp lambda_z x_P z mu')
mu = np.sqrt(2)

# Define the right-hand sides in SymPy syntax
rhs_dict = {
    "lambda_xp": 0.5 * lambda_z * (1 - z**2) / (x_P**3) * (x_P - lambda_z / sqrt(lambda_xp**2 + (1 - z**2) / (x_P**2) * lambda_z**2)),
    "lambda_z": lambda_z * z * (sqrt(z**2 - 1) / sqrt(1 - z**2) + 1 / x_P) - 0.5 * lambda_xp - 0.5 * (lambda_z**2 / x_P**2 * z) / sqrt(lambda_xp**2 + (z**2 - 1) / x_P**2 * lambda_z**2),
    "x_P": 0.5 * (z - lambda_xp / sqrt(lambda_xp**2 + (1 - z**2) / x_P**2 * lambda_z**2)),
    "z": sqrt(mu**2 - 1) * sqrt(1 - z**2) + 0.5 * (1 - z**2) / x_P**2 * (x_P - lambda_z / sqrt(lambda_xp**2 + (1 - z**2) / x_P**2 * lambda_z**2))
}

# Convert SymPy expressions to string
rhs_dict_str = {key: str(value) for key, value in rhs_dict.items()}

# Define the system of PDEs
eq = PDE(rhs_dict_str)

# Initial conditions
x_P0 = np.sqrt(mu**2 - 1) / mu
lambda_xp0 = -1 / (x_P0 - np.sqrt(mu**2 - 1) * np.sqrt(1 - x_P0**2))
lambda_z0 = 1 / (x_P0 - np.sqrt(mu**2 - 1) * np.sqrt(1 - x_P0**2))
z0 = x_P0
initial_conditions = [lambda_xp0, lambda_z0, x_P0, z0]

# Create a grid and fields
grid = UnitGrid([128, 128])
lambda_xp_field = ScalarField(grid, lambda_xp0, label="Field $\lambda_{xp}$")
lambda_z_field = ScalarField(grid, lambda_z0, label="Field $\lambda_{z}$")
x_P_field = ScalarField(grid, x_P0, label="Field $x_{P}$")
z_field = ScalarField(grid, z0, label="Field $z$")
state = FieldCollection([lambda_xp_field, lambda_z_field, x_P_field, z_field])

# Define boundary conditions
boundary_conditions = {
    'x_P': x_P_field.BC.dirichlet(1),
    'z': z_field.BC.dirichlet(0),
}

# Simulate the PDE with boundary conditions
tracker = PlotTracker(interrupts=1, plot_args={"vmin": 0, "vmax": 5})
sol = eq.solve(state, t_range=2, dt=1e-3, tracker=tracker, bc=boundary_conditions)

# Plot the results
sol.plot()
plt.show()
