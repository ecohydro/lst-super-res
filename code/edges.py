'''
    This script is used for generating edge masks from input images and saving them as PNG files.

    2022 Ryan Stofer
'''

import argparse
import os
import torch
from dataset_class import BasicDataset
import matplotlib.pyplot as plt
from utils.utils import load
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageFilter

# Define a function to parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--split', default='val', help='The split to make predictions for (train, val, or test)')
    parser.add_argument('--config', help='Path to config file', default='configs/ryan_lst_512_train.yaml')
    parser.add_argument('--threshold', help='Threshold value for edge detection', type=int, default=50)

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

    # Define the directory to save predictions
    plots_dir = os.path.join('/home/waves/projects/lst-super-res/', 'edges', str(args.split), str(args.threshold))
    input_basemap = cfg["input_basemap"]
    
    # Create the directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get the total number of samples in the dataset
    num_samples = len(dataloader)
    
    # Loop through the dataset and generate edge masks
    for sample in tqdm(dataloader, total=num_samples, desc='Plot data', unit='sample', leave=False):
        image, name, landcover = sample['image'], sample['name'], sample['landcover']
        
        # Load the RGB image
        rgb_image = load(os.path.join(input_basemap, name[0] + '.tif'), bands=3)
        
        # Convert the image to grayscale
        grayscale = rgb_image.convert("L")
        
        # Apply edge detection filter to the grayscale image
        gradient_image = grayscale.filter(ImageFilter.FIND_EDGES)
        
        # Create an edge mask using the specified threshold
        edge_mask = gradient_image.point(lambda p: p > args.threshold and 255)
        
        # Display and save the edge mask as a PNG file
        plt.imshow(edge_mask)
        plt.axis("off")
        plt.title(str(name[0] + '.tif'))
        plt.savefig(os.path.join(plots_dir, str(name[0]).split(".tif")[0] + ".png"))
        Image.open(os.path.join(plots_dir, str(name[0]).split(".tif")[0] + ".png"))
        plt.close('all')
