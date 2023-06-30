'''
    This script creates the dataset class to read and process data to feed into the dataloader. 
    The Dataset class is called for both model training and predicting in train.py and predict.py respectively.

    2022 Anna Boser
'''

import logging
import os
import operator
from os import listdir
from os.path import splitext
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
import skimage
import scipy.ndimage
from scipy import stats
import tifffile

from utils.utils import normalize_target
from utils.utils import normalize_basemap
from utils.utils import random_band_arithmetic
from utils.utils import load
from utils.utils import coarsen_image


# dataset class

class BasicDataset(Dataset):
    def __init__(self, cfg, split, predict="False"):

        self.pretrain = cfg['pretrain']
        self.toTensor = ToTensor()
        self.predict = predict
        self.split = split
        self.seed = cfg['Seed']

        self.transform = A.Compose([
                A.HorizontalFlip(p=cfg['HorizontalFlip']),
                A.VerticalFlip(p=cfg['VerticalFlip']),
                A.RandomRotate90(p=cfg['RandomRotate90'])
            ])
        self.residual = cfg['Residual']
        self.upsample_scale = cfg['upsample_scale']
        
        if self.pretrain == True:
            self.basemap_norm_loc = Path(cfg['pretrain_basemap_norm_loc'])
            self.input_basemap = Path(cfg['pretrain_basemap'])
            self.splits_loc = cfg['pretrain_splits_loc']

            if not [file for file in listdir(self.input_basemap) if not file.startswith('.')]:
                raise RuntimeError(f'No input file found in {self.input_basemap}, make sure you put your images there')

        else:
            # data paths
            self.input_target = Path(cfg['input_target'])
            self.input_basemap = Path(cfg['input_basemap'])
            self.output_target = Path(cfg['output_target'])
            self.target_norm_loc = Path(cfg['target_norm_loc'])
            self.basemap_norm_loc = Path(cfg['basemap_norm_loc'])
            self.splits_loc = cfg['splits_loc'] # location to split file to indicate which tile belongs to which split
            self.coarsen_data = cfg['coarsen_data']


            # check that images are in the given directories
            if not [file for file in listdir(self.input_basemap) if not file.startswith('.')]:
                raise RuntimeError(f'No input file found in {self.input_basemap}, make sure you put your images there')
            if not [file for file in listdir(self.input_target) if not file.startswith('.')]:
                raise RuntimeError(f'No input file found in {self.input_target}, make sure you put your images there')
            if not [file for file in listdir(self.output_target) if not file.startswith('.')]:
                raise RuntimeError(f'No input file found in {self.output_target}, make sure you put your images there')

            # normalization factors for preprocessing
            self.target_norms = pd.read_csv(self.target_norm_loc, delim_whitespace=True)

        self.basemap_norms = pd.read_csv(self.basemap_norm_loc, delim_whitespace=True).mean()

        split_files = [file for file in os.listdir(self.splits_loc) if file.endswith(".csv")]
        recent_split = sorted(split_files, key=lambda fn:os.path.getctime(os.path.join(self.splits_loc, fn)))[-1] # get most recent split
        split_file = pd.read_csv(os.path.join(self.splits_loc, recent_split))

        self.ids = [splitext(file)[0] for file in split_file[split_file['split'] == self.split]['tiles']]

        self.split_file = split_file # use this later when you get the main landcover of a tile

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    # Counts how many images are in this specified split
    def __len__(self):
        return len(self.ids)

    # Preprocess when given LST data or other input
    def preprocess(self, input_basemap_im, input_target_im, output_target_im, target_mean, target_sd):
        # turn basemap and target into tensors
        input_basemap_im = self.toTensor(input_basemap_im)*255 # the default is 0 to 255 turned into 0 to 1 -- override this since I'm doing my own normalizations
        input_target_im = self.toTensor(input_target_im) # no conversion. NA is -3.4e+38
        output_target_im = self.toTensor(output_target_im) # no conversion. NA is -3.4e+38

        # if statement here based on if we want to train on the residual of the images
        if self.residual:
            output = normalize_target(output_target_im, target_mean, target_sd, mean_for_nans=False) - normalize_target(input_target_im, target_mean, target_sd, mean_for_nans=True)
        else:
            output = normalize_target(output_target_im, target_mean, target_sd, mean_for_nans=False)
        input_target = normalize_target(input_target_im, target_mean, target_sd, mean_for_nans=True)
        ib1, ib2, ib3 = normalize_basemap(input_basemap_im, self.basemap_norms, n_bands=3)
        output_target = normalize_target(output_target_im, target_mean, target_sd, mean_for_nans=False)

        input = torch.cat([input_target, ib1, ib2, ib3], dim=0)

        return input_target, output_target, input, output
    def __getitem__(self, idx):
        name = self.ids[idx]
        input_basemap_im = list(self.input_basemap.glob(name + '.tif*'))
        assert len(input_basemap_im) == 1, f'Either no basemap input or multiple basemap inputs found for the ID {name}: {input_basemap_im}'
        if self.pretrain == True:
            input_basemap_im = load(input_basemap_im[0], bands = 3)
            input_basemap_np = np.asarray(input_basemap_im, dtype=np.float32)
            r = input_basemap_np[:,:,0]
            g = input_basemap_np[:,:,1]
            b = input_basemap_np[:,:,2]
            output_target_im, target_mean, target_sd = random_band_arithmetic(r,g,b,seed = self.seed)
            input_target_im = coarsen_image(output_target_im, self.upsample_scale)
        else:
            input_target_im = list(self.input_target.glob(name + '.tif*'))
            output_target_im = list(self.output_target.glob(name + '.tif*'))

            assert len(input_target_im) == 1, f'Either no target input or multiple target inputs found for the ID {name}: {input_target_im}'
            assert len(output_target_im) == 1, f'Either no target output or multiple target outputs found for the ID {name}: {output_target_im}'
            input_basemap_im = load(input_basemap_im[0], bands = 3)
            input_target_im = load(input_target_im[0], bands = 1)
            output_target_im = load(output_target_im[0], bands = 1)
            target_mean = self.target_norms[self.target_norms['file'] == (name + '.tif')]['mean'].mean()
            target_sd = self.target_norms[self.target_norms['file'] == (name + '.tif')]['sd'].mean()

        assert input_basemap_im.size == input_target_im.size, \
            f'Target and basemap input {name} should be the same size, but are {input_target_im.size} and {input_basemap_im.size}'
        assert input_target_im.size == output_target_im.size, \
            f'Input and output {name} should be the same size, but are {input_target_im.size} and {output_target_im.size}'

        input_target, output_target, input, output = self.preprocess(input_basemap_im, input_target_im, output_target_im, target_mean, target_sd)
        
        if self.split == "train":
            if self.predict == "False": # Removes random flipping/augmentation in predict.py
                transforms = self.transform(image=input.numpy().transpose(1,2,0), mask=output.numpy().transpose(1,2,0))
                input = transforms['image']
                output = transforms['mask']
                input, output = torch.from_numpy(input.transpose(2,0,1)), torch.from_numpy(output.transpose(2,0,1)) 

        # also get the main class of this particular image
        landcover = self.split_file[self.split_file['tiles'] == name + ".tif"]["Main Cover"]._values[0]

        return {
            'image': input, # 4 channel input: low res, RGB
            'input_target': input_target, # 1 channel low res, normalized 
            'output_target': output_target, # 1 channel high res, normalized
            'label': output, # 1 channel high res Note: either output_target OR output_target - input_target (if model trains on residual)
            'name': name, # image title
            'landcover': landcover, # landcover type
            'target_mean': target_mean, # target mean
            'target_sd': target_sd, # target standard deviation
        }


if __name__ == '__main__':
    # test code
    import argparse
    import yaml
    import random

    random.seed(1234)

    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    args = parser.parse_args()
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    dataset = BasicDataset(cfg, 'train')
    print(dataset)
    len(dataset)
    [dataset.__getitem__(i) for i in range(len(dataset))]
    dataset = BasicDataset(cfg, 'val')
    len(dataset)
    [dataset.__getitem__(i) for i in range(len(dataset))]
    dataset = BasicDataset(cfg, 'test')
    len(dataset)
    [dataset.__getitem__(i) for i in range(len(dataset))]