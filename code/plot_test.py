'''
    This script is used for generating and saving combined images from RGB input data and ground truth LST data for visualization.

    2023 Ryan Stofer
'''
# Import necessary libraries and modules
import argparse
import os
import numpy as np
import torch
from dataset_class import BasicDataset
import matplotlib.pyplot as plt
from utils.utils import load
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define a function to parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--split', default='val', help='The split to make predictions for (train, val, or test)')
    parser.add_argument('--config', help='Path to config file', default='configs/ryan_lst_512_train.yaml')

    return parser.parse_args()

# Main execution block
if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()
    
    # Print the selected split and config file
    print(f'Using Args: "{args.split}"')
    print(f'Using config "{args.config}"')
    
    # Load configuration settings from the specified YAML file
    cfg = yaml.safe_load(open(args.config, 'r'))
    loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)

    # Create a dataset for testing
    test_set = BasicDataset(cfg, args.split, predict=True)
    dataloader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the directory to save combined images
    plots_dir = os.path.join('/home/waves/projects/lst-super-res/', 'plots', str(args.split))
    input_basemap = cfg["input_basemap"]
    
    # Create the directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get the total number of samples in the dataset
    num_samples = len(dataloader)
    
    # Loop through the dataset and generate combined images
    for sample in tqdm(dataloader, total=num_samples, desc='Plot data', unit='sample', leave=False):
        image, ground_truth, name, landcover = sample['image'], sample['output_target'], sample['name'], sample['landcover']
        
        # Convert ground truth and image data to NumPy arrays
        ground_truth = np.array(ground_truth)
        image = np.array(image)
        
        # Create arrays with zeros
        zeros = np.zeros((512, 512))
        
        # Combine specific channels from the input and ground truth
        rlstb = np.stack([image[0][1], ground_truth[0][0], image[0][3]], axis=0) # R band, LST band, B band
        rlstb = np.array(np.transpose(rlstb, (1, 2, 0)))
        
        lst = np.stack([zeros, ground_truth[0][0], zeros], axis=0) # 0 band, LST band, 0 band
        lst = np.array(np.transpose(lst, (1, 2, 0)))
        
        rb = np.stack([image[0][1], zeros, image[0][2]], axis=0) # R band, 0 band, B band
        rb = np.array(np.transpose(rb, (1, 2, 0)))
        
        # Combine the above arrays horizontally
        combined_img = np.hstack((lst, rb, rlstb)) # Combined_img is not currently being plotted, but can be by simply adding plt.imshow(combined_img)
        
        # Load and display the RGB image
        rgb = np.array(load(os.path.join(input_basemap, name[0] + '.tif'), bands=3))
        plt.imshow(rgb)
        plt.axis("off")
        
        # Display the combined image with alpha transparency
        plt.imshow(lst, alpha=0.5)
        plt.axis("off")
        
        # Save the combined image as a PNG file
        plt.savefig(os.path.join(plots_dir, str(name[0]).split(".tif")[0] + ".png"))
        plt.close('all')
