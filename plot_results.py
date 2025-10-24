import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import os
import sys
import argparse
import glob
from datetime import datetime

base_path = 'mia-result/'

def main(args_folder_name: str):
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    
    # Assuming labels that the max range go from 1 to 5
    label_dict = {'WhiteMatter': 1, 'GreyMatter': 2, 'Hippocampus': 3, 'Amygdala': 4, 'Thalamus': 5}
    data = {1: [], 2: [], 3: [], 4: [], 5: []}  

    # Read .csv file
    with open(base_path + args_folder_name +'/results.csv', 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        parts = line.strip().split(';')
        label = label_dict[parts[1]] # Label is in the 2nd column
        data[label].append(float(parts[2]))  # Dice coefficient is in the 3rd column
    
    # Plot only the labels that exist in the data
    plot_data = [data[i] for i in range(1, len(data)+ 1) if data[i]]
    plot_indx = [i for i in range(1, len(plot_data)+ 1)]

    # Create boxplot
    plt.figure(figsize=(10, 6))
    bplot = plt.boxplot(plot_data, patch_artist=True)
    plt.xticks(plot_indx, [label for label, idx in label_dict.items() if data[idx]])  # Set x-t
    plt.xlabel('Labels')
    plt.xticks(rotation=35) 
    plt.ylabel('Dice Coefficient')
    plt.title('Segmentation Results', fontsize=16, fontweight='bold')

    # Add means to the boxplot
    means = [np.mean(d) for d in plot_data]

    for i, mean_val in enumerate(means):
         plt.text(i + 1, mean_val , f'{mean_val:.3f}',  # type: ignore
                 ha='center', va='center', fontweight='bold', fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='gray', alpha=0.7))
    # Add color to the boxplot 
    cmap = get_cmap('viridis') # Uniform colormap
    colors = cmap(np.linspace(0, 1, len(plot_data)))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.tight_layout() 
    plt.show()




if __name__ == '__main__':

    # Automatically find the latest folder in the mia-result directory
    latest_folder = max(
    [os.path.basename(f) for f in glob.glob(os.path.join(base_path, "*-*-*-*-*-*")) 
     if os.path.isdir(f)],
    key=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S")
    ) if glob.glob(os.path.join(base_path, "*-*-*-*-*-*")) else exit("No result folders found in 'mia-result/' directory.")
    
    # Add flag to open a specific folder
    script_dir = os.path.dirname(sys.argv[0])
    parser = argparse.ArgumentParser(description='Plot segmentation results from a specified results folder.')
    
    parser.add_argument(
        '--folder_name',
        type=str,
        default=latest_folder,
        help='Introduce folder name otherwise the latest folder will be used.'
    )

    args = parser.parse_args()

    main(args.folder_name)
