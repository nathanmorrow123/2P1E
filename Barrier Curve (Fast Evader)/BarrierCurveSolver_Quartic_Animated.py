import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import matplotlib.animation as animation

# Function to compute the quartic equation coefficients
def quartic_coeffs(x_P, x_E, y_E):
    # Coefficients of the quartic equation
    a4 = 1  # t^4 term
    a3 = -4  # t^3 term
    a2 = 2 * (1 - x_E**2 - 3*y_E**2 + x_P**2)  # t^2 term
    a1 = 4 * (x_E**2 - y_E**2 - x_P**2 + 1)  # t^1 term
    a0 = (x_E**2 + y_E**2 - x_P**2 + 1)**2 + 4 * (x_P**2 - 1) * y_E**2  # constant term
    
    return [a0, a1, a2, a3, a4]

def solve_quartic(x_P, x_E, last_y_E=None):
    if last_y_E is not None:
        if x_E < (x_P - 1):
            y_values = np.linspace(last_y_E, 1, 500)  # Scan y_E from last highest to 1.0 
        else:
            y_values = np.linspace(np.sqrt(1 - (x_P - x_E)**2), 1, 1000)  # Scan y_E from -1.0 to 1.0
    else:
        if x_E < (x_P - 1):
            y_values = np.linspace(0, 1, 1000)  # Scan y_E from -1.0 to 1.0 
        else:
            y_values = np.linspace(np.sqrt(1 - (x_P - x_E)**2), 1, 1000)  # Scan y_E from -1.0 to 1.0
    
    max_y_E = -np.inf  # Initialize with negative infinity

    for y_E in y_values:
        coeffs = quartic_coeffs(x_P, x_E, y_E)
        p = np.polynomial.Polynomial(coeffs)
        roots = p.roots()
        real_roots = [r for r in roots if np.isreal(r) and np.real(r) >= 0 and np.imag(r) == 0 and np.isclose(p(r), 0)]

        if len(real_roots) == 4:
            max_y_E = y_E

    if max_y_E == -np.inf:
        return None
        
    return max_y_E

# Function to compute and plot the barrier curve for each x_P
def plot_barrier_curve(x_P):
    x_E_values = np.linspace(0, x_P, 500)
    highest_y_E_values = []
    max_y_E = None
    for x_E in x_E_values:
        max_y_E = solve_quartic(x_P, x_E, max_y_E)
        if max_y_E is not None:
            highest_y_E_values.append(max_y_E)
        else:
            highest_y_E_values.append(np.nan)

    plt.figure(figsize=(10, 10))
    highest_y_E_values = np.array(highest_y_E_values)

    # Plot the original curve
    plt.scatter(x_E_values, highest_y_E_values, s=2, color='red')
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
    plt.title(f'Barrier Curve 2P1E for $x_P$ = {x_P:.3f}')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    # Save the plot with a unique filename
    filename = f'frames/frame_xP_{x_P:.3f}.png'
    plt.savefig(filename)
    plt.close()
    
    return filename

# Function to create animation after generating all frames
def create_animation(image_files):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    def update_frame(i):
        img = plt.imread(image_files[i])
        ax.clear()
        ax.imshow(img)
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update_frame, frames=len(image_files), repeat=True)
    
    # Save the animation as mp4
    ani.save('barrier_curve_animation.gif', dpi = 300, fps=30)

# Main function to parallelize the generation of frames
def main():
    x_P_values = np.linspace(1, 1.5, 240)  
    save_path = "frames"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Parallel processing for generating frames
    with Pool(processes=cpu_count()) as pool:
        image_files = list(tqdm(pool.imap(plot_barrier_curve, x_P_values), total=len(x_P_values)))

    # Create the animation after all frames are generated
    create_animation(image_files)

if __name__ == '__main__':
    main()
