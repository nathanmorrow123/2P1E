import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
    Helper Class for solving and understanding the quartic equation
    Nathan Morrow, 10 Dec 2024

"""


def compute_sec_min_root(mu, x_P, x_E, y_E):
    """
        
        Returns the second minimum root if four positive real roots exists
    
    """

    a4 = (mu**2-1)**2  # t^4 term
    a3 = -4 * (mu**2-1)  # t^3 term
    a2 = 2 * (2 -(mu**2-1) * (x_E**2 + y_E**2 - x_P**2 + 1) - 2*y_E**2)  # t^2 term
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)  # t^1 term
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2  # constant term
    coeffs = [a0,a1,a2,a3,a4] 
    roots = np.polynomial.Polynomial(coeffs).roots()
    real_roots = [r for r in roots if np.isreal(r) and np.real(r) >= 0 and np.imag(r) == 0]
    if len(real_roots) == 0:
        return None
    elif len(real_roots) == 4:
        sorted(real_roots)
        root_t2 = real_roots[1]
        return root_t2
    else:
        return None
    
def createBarrierData(mu,x_P,num_steps = 2000):
    """

        Creates a pandas dataframe that contains all the barrier data first quadrant only (Q2-4 assumed to be equal/mirrored)
    
    """

    all_results = []
    x_E_range = np.linspace(0,x_P,num_steps)
    for x_E in reversed(x_E_range): # x_E values are in descending order 
        y_E_range = np.linspace(0,1,num_steps) 
        results = None
        for y_E in y_E_range: 
            if np.sqrt((x_E - x_P)**2 + y_E**2) >= 1:
                t_c = compute_sec_min_root(mu, x_P, x_E, y_E)
                if t_c is not None:            
                    results = {'mu':mu,'x_P': x_P, 'x_E': x_E, 'y_E': y_E, 't_c': t_c}
                else:
                    y_E_range = np.linspace(0,y_E,num_steps)
                    break
        if results is not None:
            all_results.append(results)
    df = pd.DataFrame(all_results)
    print(df)

    return df

def plot_barrier_curve(mu,x_P,x_E_values,highest_y_E_values, mirror_values = True):
    """
        
        Creates a barrier curve plot, saves as a png (for animation/collages), and then returns the filename
    
    """
    x_E_values = np.array(x_E_values)
    highest_y_E_values = np.array(highest_y_E_values)
    plt.figure(figsize=(10, 10))
    highest_y_E_values = np.array(highest_y_E_values) # For mirroring
    
    # Plot the original curve
    plt.scatter(x_E_values, highest_y_E_values, s=2, color='red')
    
    # Mirror the barrier curve values across the four quadrants
    if mirror_values:
        plt.scatter(-x_E_values, highest_y_E_values, s=2, color='red')
        plt.scatter(-x_E_values, -highest_y_E_values, s=2, color='red')
        plt.scatter(x_E_values, -highest_y_E_values, s=2, color='red')

    plt.scatter([x_P], [0], color='blue')
    capture_radius_1 = plt.Circle((x_P, 0), 1, color='blue', fill=False)
    plt.gca().add_patch(capture_radius_1)

    plt.scatter([-x_P], [0], color='blue')
    capture_radius_2 = plt.Circle((-x_P, 0), 1, color='blue', fill=False)
    plt.gca().add_patch(capture_radius_2)

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('$x_E$')
    plt.ylabel('$y_E$')
    plt.title(f'Barrier Curve 2P1E for $\mu$ = {mu}, $x_P$ = {x_P:.4f}')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    # Save the plot with a unique filename
    filename = f'frames/frame_mu_{mu:.4f}_xP_{x_P:.4f}.png'
    plt.savefig(filename, dpi = 300, bbox_inches='tight')
    plt.close()
    
    return filename
    
def create_animation(image_files):
    """
    
        Creates a loopable animation gif using the png frames in the image_files
    
    """
    while None in image_files:
        image_files.remove(None)
    image_files = [x for x in image_files if x is not None] # Remove nones
    image_files.extend(image_files[::-1]) # Loop start finish together 

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    def update_frame(i):
        img = plt.imread(image_files[i])
        ax.clear()
        ax.imshow(img)
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update_frame, frames=len(image_files), repeat=True)
    
    # Save the animation as mp4
    ani.save('Results/barrier_curve_animation.gif', dpi = 300, fps=30)

def create_collage(image_files):
    """
    
        Creates a three by two figure collage
        image_files must be a length of six

    """
    if len(image_files)!=6:
        print("Image Files array length is not six cannot create collage")
        return
    
    fig, axs = plt.subplots(3, 2, figsize=(22, 17))
    axs = axs.flatten()

    for ax, img_file in zip(axs, image_files):
        img = plt.imread(img_file)
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('Results/Barrier_Curve_Quartic_Collage.pdf', dpi=300, bbox_inches = 'tight')
    plt.close()
