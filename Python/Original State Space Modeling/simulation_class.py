import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
# Nathan Morrow 29 January 2025

# index              0  1   2   3   4   5
# State Array:  X = [xE,yE,xP1,yP1,xP2,yP2]
# index               0        1         2
# Control Array: U = [phi,    chi,     psi] : {E,P1,P2} Respectivley
# Speed Ratio:  mu = VE / VP
# Capture States  {P1_Capture,P2_Capture,P1_Close_To_Capture,P2_Close_To_Capture}
# Goal of this program is to accuratley implement control laws into a realistic reference frame.
# Specifically for the 2P1E Fast Evader with Capture Region scenario.


def reduced_to_realistic(rot,trans,x_P,x_E,y_E,phi,chi,psi):
    
    # Origin to Origin Reverse Translation
    tempE = np.array([x_E,y_E]) - trans
    tempP1 = np.array([x_P,0]) - trans
    tempP2 = np.array([-x_P,0]) - trans
    
    # Axis Rotation 
    phi -= rot
    chi -= rot
    psi -= rot
    rot_mat = [[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]]
    tempE= rot_mat@tempE
    tempP1 = rot_mat@tempP1
    tempP2 = rot_mat@tempP2
    xE,yE = tempE
    xP1,yP1 = tempP1
    xP2,yP2 = tempP2

    return xE, yE, xP1, yP1, xP2, yP2, phi, chi, psi

def realistic_to_reduced(xE, yE, xP1, yP1, xP2, yP2,phi,chi,psi):

    # Axis Rotation
    rot = - np.arctan2(yP1-yP2,xP1-xP2)
    phi += rot
    chi += rot
    psi += rot
    rot_mat = np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
    tempE = np.array([xE,yE])
    tempP1 = np.array([xP1,yP1])
    tempP2 = np.array([xP2,yP2])
    tempE = rot_mat@tempE
    tempP1 = rot_mat@tempP1
    tempP2 = rot_mat@tempP2

    # Origin to Origin Translation
    trans = [-(tempP1[0]-0.5*(tempP1[0]-tempP2[0])),-(tempP1[1])]
    tempE += trans
    tempP1 += trans
    tempP2 += trans
    x_P = tempP1[0]
    x_E,y_E = tempE
    return rot,trans,x_P,x_E,y_E,phi,chi,psi

def compute_sec_min_root(mu, x_P, x_E, y_E):
    """
        
        Returns the second minimum root if four positive real roots exists
    
    """

    a4 = (mu**2-1)**2  # t**4 term
    a3 = -4 * (mu**2-1)  # t**3 term
    a2 = 2 * (2 -(mu**2-1) * (x_E**2 + y_E**2 - x_P**2 + 1) - 2*y_E**2)  # t**2 term
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)  # t**1 term
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2  # constant term
    coeffs = [a0,a1,a2,a3,a4] 
    roots = np.polynomial.Polynomial(coeffs).roots()
    real_roots = [r for r in roots if np.isreal(r) and np.real(r) >= 0 and np.imag(r) == 0]
    if len(real_roots) == 0:
        return None
    elif len(real_roots) == 1:
        return None
    elif len(real_roots) == 2:
        t_c = sorted(np.real(real_roots))[1]
        return t_c
    elif len(real_roots) == 3:
        return None
    elif len(real_roots) == 4:
        t_c = sorted(np.real(real_roots))[1]
        return t_c
    else:
        return None
    
def compute_quartic_headings(t,mu,x_P,x_E,y_E):
    if y_E == 0:
        y = np.sqrt(mu**2 * t**2 - x_E**2)
        y = np.sqrt((t + 1)**2 - x_P**2)
    else:
        y = (x_E**2 + y_E**2 - x_P**2 + 1 + 2 * t - (mu**2 - 1) * t**2) / (2 * y_E)

    print(t,mu,x_P,x_E,y_E)

    # Triangle one P1 - Origin - Intercept
    chi = np.arctan2(y,-x_P)

    # Triangle two Evader - EvaderYaxisProj - Intercept
    if np.isclose(x_E,0):
        if y > 0:
            phi = np.pi / 2
        else:
            phi = -np.pi / 2
    else:
        phi = np.arctan2(y-y_E,-x_E)

    psi = np.pi -chi 

    return phi,chi,psi

class PursuitSimulation:
    def __init__(self, speed_ratio, E_initial, p1_initial, p2_initial, control = "surf", continue_after_capture = False):
        self.mu = speed_ratio
        self.X = np.array([*E_initial,*p1_initial, *p2_initial])
        self.t = 0.0
        self.U = np.array([np.pi/2, np.pi/2, np.pi/2])
        self.MoE = 1e-2
        self.control = control
        self.CaC = continue_after_capture
        self.capture_states = [False, False,False,False]
        self.capture_buffer_counter = 0
        self.trail_length = 10
        self.pursuer_trails = [[], []]
        self.evader_trail = []
        
        self.init_plot()
        self.init_players()
    
    def init_plot(self):
        #plt.style.use('dark_background')
        plt.rcParams["font.family"] = "Times New Roman"
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0, 20)
        plt.title(f'Two Cutters and Fugitive Ship Problem, Time: {np.round(self.t,4)}')
        plt.xlabel('$X$')
        plt.ylabel('$Y$')

    def init_players(self):
        xE, yE, xP1, yP1, xP2, yP2 = self.X
        self.pursuers, = self.ax.plot([], [], 'bo', markersize=5, alpha=1)
        self.evader, = self.ax.plot([], [], 'ro', markersize=5, alpha=1)
        self.capture_circles = [
            Circle((xP1, yP1), 1, linestyle='--', linewidth=1, edgecolor='darkblue', facecolor='none'),
            Circle((xP2, yP2), 1, linestyle='--', linewidth=1, edgecolor='darkblue', facecolor='none')
        ]
        for circle in self.capture_circles:
            self.ax.add_patch(circle)
    
    def update_plot_limits(self):
        x_vals, y_vals = self.X[::2], self.X[1::2]
        x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
        y_min, y_max = min(y_vals) - 1, max(y_vals) + 1
        data_range = max(x_max - x_min, y_max - y_min)
        self.ax.set_xlim((x_min, x_min + data_range))
        self.ax.set_ylim((y_min, y_min + data_range))
        plt.title(f'Two Cutters and Fugitive Ship Problem, Time: {self.t:.2f}')

    def no_contact_pure_pursuit_control(self):
        
        xE, yE, xP1, yP1, xP2, yP2 = self.X
        
        phi = np.pi/2  # Evader moves straight for now (NOT CORRECT)
        chi = np.arctan2(yE - yP1, xE - xP1)  # P1 pursues evader
        psi = np.arctan2(yE - yP2, xE - xP2)  # P2 pursues evader
        
        self.U = np.array([phi, chi, psi])
        
        print("no_contact_PP")
    
    def contact_sub_optimal(self):
        phi,chi,psi = self.U
        xE, yE, xP1, yP1, xP2, yP2 = self.X
        rot,trans,x_P,x_E,y_E,phi,chi,psi = realistic_to_reduced(xE, yE, xP1, yP1, xP2, yP2,phi,chi,psi)
        
        # P1 engages in "Pure Pursuit"
        chi = np.arctan2(y_E,-(x_P-x_E))
        
        # E Surfs P1
        alpha = np.arccos((1-self.MoE)/self.mu)
        phi = chi - alpha
        
        # P2 Mirrors
        psi = np.pi - chi
        
        xE, yE, xP1, yP1, xP2, yP2,phi,chi,psi = reduced_to_realistic(rot,trans,x_P,x_E,y_E,phi,chi,psi)

        self.U = np.array([phi,chi,psi])
        
        print("contact_sub_optimal")

    def contact_optimal(self):
        phi,chi,psi = self.U
        xE, yE, xP1, yP1, xP2, yP2 = self.X
        rot,trans,x_P,x_E,y_E,phi,chi,psi = realistic_to_reduced(xE, yE, xP1, yP1, xP2, yP2,phi,chi,psi)
        
        # P1 engages in "Pure Pursuit"
        chi = np.arctan2(y_E,-(x_P-x_E))
        
        

        # E Surfs P1
        alpha = np.arccos((1-self.MoE)/self.mu)
        phi = chi - alpha

        xE, yE, xP1, yP1, xP2, yP2,phi,chi,psi = reduced_to_realistic(rot,trans,x_P,x_E,y_E,phi,chi,psi)

        self.U = np.array([phi,chi,psi])
        print("contact_optimal")
        
    def collision_course_control(self):
        xE, yE, xP1, yP1, xP2, yP2 = self.X
        phi,chi,psi = self.U
        mu = self.mu

        rot,trans, x_P,x_E,y_E,phi,chi,psi = realistic_to_reduced(xE, yE, xP1, yP1, xP2, yP2,phi,chi,psi)
        
        t_c = compute_sec_min_root(mu, x_P, x_E, y_E)

        if t_c is not None:
            phi,chi,psi = compute_quartic_headings(t_c,mu,x_P,x_E,y_E)
        
        else:
            self.no_contact_pure_pursuit_control()
            print("Capture is not achievable switching to pure pursuit!")
            return
        
        xE, yE, xP1, yP1, xP2, yP2,phi,chi,psi = reduced_to_realistic(rot,trans, x_P,x_E,y_E,phi,chi,psi)
        self.X = xE, yE, xP1, yP1, xP2, yP2
        phi -= rot
        chi -= rot
        psi -= rot
        self.U = np.array([phi, chi, psi])
        
        print("no_contact_CC")
    
    def propagate_players(self):
        if self.control == "surf":
            if ( self.capture_states[2] or self.capture_states[3]):
                self.contact_sub_optimal()
            else:
                self.collision_course_control()
        elif self.control == "collision_course":
            self.collision_course_control()
        elif self.control == "pure_pursuit":
            self.no_contact_pure_pursuit_control()
        else:
            print("Invalid Control Strategy, Switching to Surf")
            self.control = "surf"

        phi, chi, psi = self.U
        vE, vP, dt = self.mu, 1.0, .01
        self.X = np.array(self.X)

        self.X[0:2] += (vE * dt) * np.array([np.cos(phi), np.sin(phi)])  # Evader Motion
        self.X[2:4] += (vP * dt) * np.array([np.cos(chi), np.sin(chi)])  # P1 Motion
        self.X[4:6] += (vP * dt) * np.array([np.cos(psi), np.sin(psi)])  # P2 Motion
        self.t += dt
        self.evader_trail.append([self.X[0], self.X[1]])
        self.pursuer_trails[0].append([self.X[2], self.X[3]])
        self.pursuer_trails[1].append([self.X[4], self.X[5]])
        
        if len(self.evader_trail) > self.trail_length:
            self.evader_trail.pop(0)
        if len(self.pursuer_trails[0]) > self.trail_length:
            self.pursuer_trails[0].pop(0)
        if len(self.pursuer_trails[1]) > self.trail_length:
            self.pursuer_trails[1].pop(0)

    def checkCapture(self):
        P1_Capture_State,P2_Capture_State,P1_Close_To_Capture,P2_Close_To_Capture = self.capture_states
        
        xE, yE, xP1, yP1, xP2, yP2 = self.X
        if (np.sqrt((xE-xP1)**2+(yE-yP1)**2)<1):
            P1_Capture_State = True
        else:
            P1_Capture_State = False
        if (np.sqrt((xE-xP2)**2+(yE-yP2)**2)<1):
            P2_Capture_State = True
        else:
            P2_Capture_State = False

        if (np.sqrt((xE-xP1)**2+(yE-yP1)**2)<1 + self.MoE):
            P1_Close_To_Capture = True
            print("Close To Capture!")
        if (np.sqrt((xE-xP2)**2+(yE-yP2)**2)<1 + self.MoE):
            P2_Close_To_Capture = True
            print("Close To Capture!")
        
        self.capture_states = [P1_Capture_State,P2_Capture_State,P1_Close_To_Capture,P2_Close_To_Capture]
        
    def update(self, frame):
        self.pursuers.set_data(self.X[2::2], self.X[3::2])
        self.evader.set_data(self.X[0], self.X[1])
        
        for i, circle in enumerate(self.capture_circles):
            circle.center = (self.X[2 + i * 2], self.X[3 + i * 2])
        
        for i in range(2):
            if self.pursuer_trails[i]:
                trail_x, trail_y = zip(*self.pursuer_trails[i])
                self.ax.plot(trail_x, trail_y, 'b-', alpha=0.2)
        
        if self.evader_trail:
            evader_x, evader_y = zip(*self.evader_trail)
            self.ax.plot(evader_x, evader_y, 'r-', alpha=0.2)
        
        if (self.capture_states[0] or self.capture_states[1]):
            print("captured")
            self.capture_buffer_counter += 1
            if not(self.capture_states[0] and self.capture_states[1]): # Only continue after capture if evader isn't captured by both 
                if self.CaC: # Continue after Capture setting
                    self.propagate_players()
                    self.checkCapture()
                    self.update_plot_limits()
            return [self.pursuers, self.evader] + self.capture_circles
        else:
            self.propagate_players()
            self.checkCapture()
            self.update_plot_limits()
            return [self.pursuers, self.evader] + self.capture_circles
            
    def run_simulation(self):
        if not os.path.exists("Results"):
            os.makedirs("Results")
        
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=250, blit=False)
        print("Saving animation as a GIF...")
        self.ani.save('Results/pursuers_and_fugitive.gif', writer='pillow', fps=30)
        plt.savefig('last_frame.png', dpi = 300)






