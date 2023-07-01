'''
    This script performs predictions using the trained UNet model and computes evaluation metrics (R2, RMSE, SSIM scores) on desired split. 
    Optionally, can generate visualization plots for each prediction.
    
    2022 Anna Boser
'''
import argparse
import logging
import os
import time
from tkinter import image_types
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset_class import BasicDataset
from unet import UNet

from utils.utils import unnormalize_target
from utils.utils import load
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from predict_vis import predict_vis

def predict_img(net,
                dataloader,
                device, 
                basemap_norms, 
                predictions_dir,
                residual,
                visualize,
                input_basemap,
                split,
                experiment_dir):

    net.eval()
    num_images = len(dataloader)

    # Initialize metrics data frame to store all metric info
    metrics_df = pd.DataFrame(columns=['file', 'landcover', 'r2_pred', 'mse_pred', 'mae_pred','ssim_pred','r2_coarse', 'mse_coarse', 'mae_coarse', 'ssim_coarse'])

    # iterate over chosen split set
    for image in tqdm(dataloader, total=num_images, desc='Making predictions', unit='image', leave=False):
        image, name, target_input, name, landcover = image['image'], image['name'], image['input_target'], image['name'], image['landcover']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        target_input = target_input.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred = net(image)
            
            for i, output in enumerate(pred):
                logging.info(f'\nPredicting image {name} ...')

                # put predictions back into native format
                # if we are calculating original values keep unnormalize_target
                # else if we are calculating residuals, add to coarse array
                if residual:
                    output = unnormalize_target(target_input+output, image['target_mean'], image['target_sd']) # image is a float 32 torch
                else:
                    output = unnormalize_target(output, image['target_mean'], image['target_sd'])

                # un-tensorify ouput prediction image
                output = output.cpu().numpy()
                output = output.squeeze()
                output_image = Image.fromarray(output)

                # save prediction
                out_filename = os.path.join(predictions_dir, name[i] + ".tif")
                output_image.save(out_filename)

                logging.info(f'Predictions saved to {out_filename}')

                # call input and ground truth label for evaluating metrics
                input_target = unnormalize_target(image['input_target'], image['target_mean'], image['target_sd'])
                input_target = input_target.cpu().numpy()
                input_target = input_target.squeeze()

                label = unnormalize_target(image['output_target'], image['target_mean'], image['target_sd'])
                label = label.cpu().numpy()
                label = label.squeeze()

                # Masking for normal training
                if not pretrain:
                    mask = ground_truth < 0
                    output[mask] = np.nan
                    label[mask] = np.nan
                    coarse[mask] = np.nan

                # Compute and save input target and prediction metrics as a .csv file
                pred_metrics = predict_vis(output, input_target, label, name, landcover, split, input_basemap, visualize, experiment_dir)

                # Append prediction metric to base data frame
                metrics_df = metrics_df.append(pred_metrics, ignore_index = True)

        # save the pandas dataframe of evaluation metrics as csv
        os.makedirs(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(split)), exist_ok = True)
        metrics_df.to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(split), "prediction_metrics.csv"))

        # Save the average metrics for entire dataset
        metrics_df.mean().to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(split), "prediction_metrics_mean.csv"))
    return 


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    parser.add_argument('--model_epoch', '-m', default='last', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--split', default='val', help='The split to make predictions for (train, val, or test)')
    parser.add_argument('--visualize', default=True, help='Visualize plots for each prediction')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)
    
    print(f'Using Args: "{args.split}"')

    random.seed(cfg['Seed'])

    val_set = BasicDataset(cfg, args.split, predict="True")
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    net = UNet(n_channels=4, n_classes=1, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    if args.model_epoch == 'last':
        model_epoch = cfg['epochs']
    else:
         model_epoch = args.model_epoch
    logging.info(f'Loading model checkpoints/checkpoint_epoch{model_epoch}.pth')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(cfg['experiment_dir'], f'checkpoints/checkpoint_epoch{model_epoch}.pth'), map_location=device))

    logging.info('Model loaded!')

    # get the directory to save predictions to
    predictions_dir = os.path.join(cfg['experiment_dir'], 'predictions', str(args.split))
    os.makedirs(predictions_dir, exist_ok=True)

    pretrain = cfg['pretrain']
    if pretrain:
        basemap_norm_loc = Path(cfg['pretrain_basemap_norm_loc'])
        input_basemap = cfg["pretrain_basemap"]
    else:
        basemap_norm_loc = Path(cfg['basemap_norm_loc'])
        input_basemap = cfg["input_basemap"]
    basemap_norms = pd.read_csv(basemap_norm_loc, delim_whitespace=True).mean()
    # See if we calculate the residual instead
    residual = cfg['Residual']
    start = time.time()
    predict_img(net, val_loader, device, basemap_norms, predictions_dir, residual, args.visualize, input_basemap, args.split, cfg['experiment_dir'])
    end = time.time()-start
    print('Total prediction time:', end)
