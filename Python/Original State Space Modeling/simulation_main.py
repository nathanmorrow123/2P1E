import simulation_class as sc
import numpy as np


"""# Speed Ratio VE/VP
mu = np.sqrt(2)

# Initial Positions
xE = 0
yE = 0
xP1 = 3
yP1 = 0
xP2 = -3
yP2 = 0

sc.run_sim(mu,(xE,yE),(xP1,yP1),(xP2,yP2))
"""
sim = sc.PursuitSimulation(speed_ratio=1.1, E_initial=np.array([0.02,0.6]), p1_initial=np.array([1.1, 0]), p2_initial=np.array([-1.1, 0]))
sim.run_simulation()
