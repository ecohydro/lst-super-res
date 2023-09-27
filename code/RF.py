'''
    This script enhances coarsened LST images using a Random Forest regressor.

    2022 Ryan Stofer
'''
import argparse
import logging
import os
import time
from tkinter import image_types

import numpy as np
from PIL import Image

from dataset_class import BasicDataset

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from utils.utils import normalize_target

# Add get arguments function
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    parser.add_argument('--model_epoch', '-m', default='last', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--split', default='val', help='The split to make predictions for (train, val, or test)')

    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()

    args = get_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)

    print(f'Using Args: "{args.split}"')
    data_set = BasicDataset(cfg, args.split, predict="True") 
    data_loader = DataLoader(data_set, shuffle=False, drop_last=True, **loader_args)

    num_data_batches = len(data_loader)

    output_target = cfg["output_target"]
    metrics_df = pd.DataFrame(columns=['file', 'landcover', 'r2_pred', 'rmse_pred']) # Create an empty DataFrame to store prediction metrics

    rmse = np.zeros(shape= len(data_loader))
    r2 = np.zeros(shape= len(data_loader))
    i = 0
    for batch in tqdm(data_loader, total=num_data_batches, desc='Making predictions', unit='batch', leave=False):
        # Get each individual image data and metadata
        image, name, target_input, target_mean, target_sd, landcover = batch['image'], batch['name'], batch['input_target'], batch['target_mean'], batch['target_sd'], batch['landcover']

        # convert for CPU usage
        image = image.cpu().numpy()
        image = image.squeeze()

        target_mean = target_mean.cpu().numpy()
        target_sd = target_sd.cpu().numpy()
        # Initialize data frame with coarsened image along with RGB and ground truth values
        coarse = image[0]
        ground_truth = np.array(Image.open(os.path.join(output_target, ''.join(name) + ".tif"))).flatten()
        ground_truth = ground_truth[~np.isnan(ground_truth)]
        ground_truth = normalize_target(ground_truth, target_mean, target_sd, mean_for_nans=False)
        R = image[1]
        G = image[2]
        B = image[3]
        df = pd.DataFrame(data = {'coarse' : coarse.flatten(), 'R': R.flatten(), 'G': G.flatten(), 'B': B.flatten(), 'ground_truth': ground_truth})
        df = df.dropna()

        # Fits regressor to image
        regressor = RandomForestRegressor()
        regressor.fit(df.iloc[:,1:4],df.iloc[:,0])
        y_pred = regressor.predict(df.iloc[:,1:4])
        
        # Compute relevant metrics after unnormalizing prediction and ground truth
        ground_truth = df['ground_truth']
        y_pred = (y_pred*target_sd) + target_mean
        ground_truth = (ground_truth*target_sd) + target_mean

        r2_pred = round(metrics.r2_score(ground_truth[~np.isnan(ground_truth)], y_pred[~np.isnan(y_pred)]), 2)
        rmse_pred = round(np.sqrt(metrics.mean_squared_error(ground_truth[~np.isnan(ground_truth)], y_pred[~np.isnan(y_pred)])), 2)

        r2_coarse = round(metrics.r2_score(ground_truth[~np.isnan(ground_truth)], coarse[~np.isnan(coarse)]), 2)
        rmse_coarse = round(np.sqrt(metrics.mean_squared_error(ground_truth[~np.isnan(ground_truth)], coarse[~np.isnan(coarse)])), 2)

        # Generate dataframe to save results
        df2 = pd.DataFrame({
        'file': [name], 
        'landcover': [landcover],   
        'r2_pred': [r2_pred],
        'rmse_pred': [rmse_pred],
        'r2_coarse': [r2_coarse],
        'rmse_coarse': [rmse_coarse]
        })
        metrics_df = metrics_df.append(df2, ignore_index = True)

    # Save dataframe as .csv file
    metrics_df.to_csv(os.path.join('RF','results.csv')) # Save the evaluation metrics dataframe to a CSV file
    metrics_df.groupby('landcover').mean().to_csv(os.path.join('RF','lc_results.csv'))
    end = time.time()-start
    print('End Time:', end) # Print the time taken for the prediction process to complete