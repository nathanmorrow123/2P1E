import simulation_class as sc
import numpy as np

"""

Nathan Morrow, Febuary 3rd 2025
Runner File, for the 2P1E simulation_class.py

Only works for interceptable trajectory scenarios
See Barrier Curves to selct initial points
Initial Coords are in the realistic frame
Controls are calculated in a reduced frame

Control options are:
Collision Course (Straight line trajectories, like PN, uses quartic to find headings), "collision_course"
Pure Pursuit (Every instance Pursuers aim towards the Evader), "pure_pursuit"
Surf (Collision Course up untill contact, then surf the edge of closest Pusuers Capture Region towards Reduced Y axis), "surf"

"""

# Speed Ratio: mu = VE/VP
mu = 1.1
Xp0 = (mu/np.sqrt(mu**2-1)) - 0.1
Xe0 = (mu/np.sqrt(mu**2-1)) - 1.0 - 0.1
Ye0 = 0.0

sim = sc.PursuitSimulation(speed_ratio=1.1, E_initial=np.array([1.3,0]), p1_initial=np.array([2.4, 0]), 
                            p2_initial=np.array([-2.4, 0]), control = "surf", continue_after_capture = True)

sim.run_simulation()
