import numpy as np
import pandas as pd
import quartic_class as qc

"""

    Runner main file to create collages/frames/animations of the 2P1E DG 
    Capture Region/ Barrier Region
    Nathan Morrow 11 Dec 2024

"""


# Define the various x_P values and mu values
mu_x_P_cases = [
   #|   mu     |  x_P    |
   (np.sqrt(2), np.sqrt(2)),
   (np.sqrt(2),       1.1),
   (np.sqrt(2),      1.45),
   (1.1,              1.1),
   (1.1,             1.05),
   (1.1,                3)
]

def main():
    total_results = pd.DataFrame([])
    image_files = []
    for mu, x_P in mu_x_P_cases:
        results_df = qc.createBarrierData(mu,x_P)
        results_df.to_csv(f'Results/data_table_mu_{mu}_x_P{x_P}.txt', sep='\t', index=False)
        file = qc.plot_barrier_curve(mu,x_P,results_df['x_E'].tolist(),results_df['y_E'].tolist(),results_df['t_c'].tolist())
        image_files.append(file)
        pd.concat([total_results,results_df])
    # Creates a 3x2 Collage with our six mu x_P cases
    qc.create_collage(image_files)
    # Write DataFrame data to a text file 
    total_results.to_csv(f'Results/data_table_all.txt', sep='\t', index=False)

if __name__ == '__main__':
    main()
