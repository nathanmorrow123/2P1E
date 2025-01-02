import numpy as np
import matplotlib.pyplot as plt

"""

    helper class to create cartesian ovals in the 2P1E fast evader DG with capture circles
    nathan morrow 11 dec 2024
    
"""
def getCartesianOvalData(x_P,x_E,y_E,mu):
     if y_E == 0:
        # Args ( Pursurer starting x coord, Evader starting coord, Speed ratio Ve/Vp)        
        t_min = (1/(1+mu))*(x_P-x_E-1)
        numerator = np.sqrt((-2 * x_P + 2 * x_E - 2)**2 - 4 * (mu**4 - 2 * mu**2 + 1) * (2 * x_E * x_P - x_P**2 - 2 * x_P - x_E**2 + 2 * x_E - 1)) + 2 * x_P - 2 * x_E + 2
        denominator = 2 * (mu**4 - 2 * mu**2 + 1)
        t_max_true = numerator/denominator

        # Generate parameter t
        t = np.linspace(t_min, t_max_true, 100000, endpoint=True)

        # Parametric equations
        x1 = [x_P+((mu**2-1)*t**2-2*t-(x_P-x_E)**2-1)/(2*(x_P-x_E)),x_P+((mu**2-1)*t**2-2*t-(x_P-x_E)**2-1)/(2*(x_P-x_E))]
        y1 = [np.sqrt((t+1)**2-(x1[0]-x_P)**2),-np.sqrt((t+1)**2-(x1[1]-x_P)**2)] # Two values for y a positive and a neqative sqrt
        x2 = [-x_P-((mu**2-1)*t**2-2*t-(x_P-x_E)**2-1)/(2*(x_P-x_E)),-x_P-((mu**2-1)*t**2-2*t-(x_P-x_E)**2-1)/(2*(x_P-x_E))]
        y2 = [np.sqrt((t+1)**2-(x2[0]+x_P)**2),-np.sqrt((t+1)**2-(x2[1]+x_P)**2)] # Two values for y a positive and a neqative sqrt
        return (x1,y1,x2,y2)
     
     else:
    
        # State space translation (x_P,x_E,y_E,mu) - > (d1,d2,theta1,theta2)
        d1 = np.sqrt((x_P-x_E)**2+y_E**2)
        d2 = np.sqrt((x_P+x_E)**2+y_E**2)
        theta1 = np.arcsin(y_E/d1)
        theta2 = np.arcsin(y_E/d2)
        
        ## First Pursuer Cartesian Oval
        # Generate parameter t
        t_min = (d1-1)/(mu+1)
        t_max = (d1+1)/(mu-1)
        t = np.linspace(t_min, t_max, 100000, endpoint=True)
        x1 = [((mu**2-1)*t**2-2*t-(d1-1)**2)/(2*d1),((mu**2-1)*t**2-2*t-(d1-1)**2)/(2*d1)]
        y1 = [np.sqrt((t+1)**2-(((mu**2-1)*t**2-2*t-d1**2-1)/(2*d1))**2),-np.sqrt((t+1)**2-(((mu**2-1)*t**2-2*t-d1**2-1)/(2*d1))**2)] # Two values for y a positive and a neqative sqrt

        ## Rotation
        # Rotate (x1,y1) about P1 by theta1
        cos_var = np.cos(theta1)
        sin_var = np.sin(theta1)
        x1 = np.array(x1)
        x1 = np.hstack((x1[0],x1[1]))
        y1 = np.array(y1)
        y1 = np.hstack((y1[0],y1[1]))
        x1 = x1 - x_P # Translate
        coords = np.vstack((x1,y1))
        print(coords.shape)
        rotation_mat = np.array([[cos_var,sin_var],[-sin_var,cos_var]]) # Clockwise rotation mat
        rotated_coords = rotation_mat@coords # Rotate
        x1,y1 = rotated_coords
        print(rotated_coords.shape)
        x1 = x1 + x_P # Translate

        ## Second Pusuer Cartesian Oval
        # Generate parameter t
        t_min = (d2-1)/(mu+1)
        t_max = (d2+1)/(mu-1)
        t = np.linspace(t_min, t_max, 100000, endpoint=True)
        x2 = [-((mu**2-1)*t**2-2*t-(d2-1)**2)/(2*d2),-((mu**2-1)*t**2-2*t-(d2-1)**2)/(2*d2)]
        y2 = [np.sqrt((t+1)**2-(((mu**2-1)*t**2-2*t-d2**2-1)/(2*d2))**2),-np.sqrt((t+1)**2-(((mu**2-1)*t**2-2*t-d2**2-1)/(2*d2))**2)] # Two values for y a positive and a neqative sqrt

        # Rotate (x2,y2) about P2 by theta2
        cos_var = np.cos(theta2)
        sin_var = np.sin(theta2)
        x2 = np.array(x2)
        x2 = np.hstack((x2[0],x2[1]))
        y2 = np.array(y2)
        y2 = np.hstack((y2[0],y2[1]))
        x2 = x2 + x_P # Translate
        coords = np.vstack((x2,y2))
        rotation_mat = np.array([[cos_var,-sin_var],[sin_var,cos_var]]) # CCW Rotation mat
        rotated_coords = rotation_mat@coords # Rotate
        x2,y2 = rotated_coords
        x2 = x2 - x_P # Translate

        return (x1,y1,x2,y2)
         
    

def plotCartesianOval(x_P,x_E,y_E,mu):
    x1,y1,x2,y2 = getCartesianOvalData(x_P,x_E,y_E,mu)

    ## Plot the Cartesian oval
    # Set Figure Size
    plt.figure(figsize=(6, 4))
    # Set the aspect of the plot to be equal
    
    #plt.style.use('dark_background')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True  # Enable LaTeX rendering
    plt.scatter(x1, y1, s=1, label='P1 Cartesian Oval')
    plt.scatter(x2, y2, s=1, label = 'P2 Cartesian Oval')
    
    # Plot the players
    plt.plot(x_P, 0, '.', label='(Pursuer 1)')
    plt.plot(-x_P, 0, '.', label='(Pursuer 2)')
    plt.plot(x_E, y_E,  '.', label ='(Evader)')
    
    plt.gca().set_aspect('equal', adjustable='box')
    # Labels and title
    plt.xlabel(rf"$x_E$")
    plt.ylabel(rf"$y_E$")
    plt.title(rf"2P1E Cartesian $\textit{{oval}},\mu = {mu:.4} $")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.set_cmap('hot')
    plt.savefig(rf'Results/CO_2P1E_mu_{mu:.4}_x_P_{x_P:.4}_x_E_{x_E:.4}_y_E_{y_E:.4}.png',dpi=600)


