'''
    Visualize model outputs:
    This script loads in either val or test data and creates predictions using the trained model of choice. 
    These predictions are plotted and evaluated using MSE, SSIM, and R2_score metrics. 

    2022 Anna Boser
'''

import os
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import get_r2
from utils.utils import get_mse
from utils.utils import get_mae
from utils.utils import get_ssim
from utils.utils import load


def predict_vis(pred, coarse, ground_truth, image, landcover, split, input_basemap, visualize, experiment_dir):
    # Compute metrics
    r2_pred = get_r2(ground_truth, pred)
    mse_pred = get_mse(ground_truth, pred)
    ssim_pred = get_ssim(ground_truth, pred)
    mae_pred = get_mae(ground_truth, pred)
    ssim_coarse = get_ssim(ground_truth, coarse)
    r2_coarse = get_r2(ground_truth, coarse)
    mse_coarse = get_mse(ground_truth, coarse)
    mae_coarse = get_mae(ground_truth, coarse)

    # Create data frame observation
    df = pd.DataFrame({
        'file': [image[0] + '.tif'], 
        'landcover': [landcover[0]], 
        'r2_pred': [r2_pred], 
        'mse_pred': [mse_pred], 
        'mae_pred': [mae_pred],
        'ssim_pred': [ssim_pred], 
        'r2_coarse': [r2_coarse], 
        'mse_coarse': [mse_coarse], 
        'mae_coarse': [mae_coarse],
        'ssim_coarse': [ssim_coarse]
        })

    # Visualize image observation and metrics (optional)
    if visualize:
        # Load in basemap
        rgb = np.array(load(os.path.join(input_basemap, image[0] + '.tif'), bands = 3))

        # Visualize all four images with associated metrics
        plt.figure(figsize=(24,6))
        plt.subplot(1, 4, 1)
        plt.imshow(ground_truth, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
        plt.title(str(image[0] + '.tif'))
        plt.axis("off")
        plt.subplot(1, 4, 2)
        plt.imshow(pred, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
        plt.title(f'Prediction: r2 = {str(r2_pred)}, mse = {str(mse_pred)}, mae = {str(mae_pred)}')
        plt.axis("off")
        plt.subplot(1, 4, 3)
        plt.imshow(coarse, vmin = np.nanmin(coarse), vmax= np.nanmax(coarse), cmap = 'coolwarm') 
        plt.title(f'Coarsened input: r2 = {str(r2_coarse)}, mse = {str(mse_coarse)}, mae = {str(mae_coarse)}')
        plt.axis("off")
        plt.subplot(1, 4, 4)
        plt.imshow(rgb)
        plt.title(str(landcover[0]))
        plt.axis("off")

        # create a directory to store prediction plots
        os.makedirs(os.path.join(experiment_dir, "prediction_plots", str(split)), exist_ok=True)
        plt.savefig(os.path.join(experiment_dir, "prediction_plots", str(split), str(image[0]).split(".tif")[0]+".png"))
        # plt.show() # for the notebook version
        plt.close('all')
    return df