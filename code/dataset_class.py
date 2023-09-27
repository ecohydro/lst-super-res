'''
    This script creates the dataset class to read and process data to feed into the dataloader. 
    The Dataset class is called for both model training and predicting in train.py and predict.py respectively.

    2022 Anna Boser
'''

import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
import random

from utils.utils import normalize_target
from utils.utils import normalize_basemap
from utils.utils import random_band_arithmetic
from utils.utils import load
import math
#from utils.utils import coarsen_image

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def coarsen_image(image, factor):
    """
    Takes a single band PIL image and 
    (1) coarsens it by factor using box resampling and 
    (2) resamples it back to its original resolution using nearest neighbor interpolation

    If the number of pixels does not evenly divide into the number of pixels of the origial image, 
    there will be "partial pixels" along the right and bottom edges of the image. 

    Returns: the coarsened and resmapled to original resolution PIL image. 

    """
    # Convert the image to a numpy array and create a copy of the image array
    image_array = np.copy(np.asarray(image))
    
    # Locate nan values and make them NaN if they are not already (-3.3999999521443642e+38 is NaN)
    #image_array[image_array == -3.3999999521443642e+38] = np.nan
    
    # Make a mask of NaN values in the image array
    nan_indices = np.isnan(image_array)
    
    # Calculate the new dimensions based on the coarsening factor
    new_width = math.ceil(image.width / factor)
    new_height = math.ceil(image.height / factor)

    # Iterate over each new pixel and compute the average of valid pixels
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the indices of the corresponding pixels in the original image
            start_y = i * factor
            end_y = (i + 1) * factor
            start_x = j * factor
            end_x = (j + 1) * factor
    
            # Compute the average of valid pixels within the new pixel
            valid_pixels = image_array[start_y:end_y, start_x:end_x][~nan_indices[start_y:end_y, start_x:end_x]]
            average_value = np.mean(valid_pixels) if valid_pixels.size > 0 else np.nan
    
            # Assign the average value to the corresponding pixels in the coarsened array
            image_array[start_y:end_y, start_x:end_x] = average_value
            
    # Convert the coarsened array back to an image
    coarsened_image = Image.fromarray(image_array)

    # Scale back up to the original size using nearest neighbor interpolation
    coarsened_image = coarsened_image.resize((image.width, image.height), resample=Image.Resampling.NEAREST)
    return coarsened_image


def shade(image):
    random.seed()
    # Get image dimensions
    width, height, channels = image.shape

    # Define shading factor between 0 and 1
    shading_factors = [random.uniform(0.5, 0.75) for _ in range(channels)] # for example, to reduce the brightness by 50% to 100%

    # Define the number of vertices of the polygon
    num_vertices = random.randint(3, 10)

    # Generate random vertices
    vertices = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_vertices)]

    # Create a new image with the same size as the original
    mask = Image.new('L', (width, height), color=0)

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Fill the polygon on the new image
    draw.polygon(vertices, fill=255)

    # Convert the new image to a NumPy array for manipulation
    mask = np.array(mask)

    for channel in range(channels):
        image[:, :, channel][mask>0] = (image[:, :, channel][mask>0] * shading_factors[channel]).astype(np.uint8)
    return(image)

class BasicDataset(Dataset):
    """Dataset class for image data.

    Args:
        cfg (dict): Configuration dictionary.
        split (str): Split type ("train", "val", "test").
        predict (bool, optional): Flag indicating if the dataset is used for prediction. Defaults to False.
    """
    
    def __init__(self, cfg: dict, split: str, predict: bool = False):

        self.pretrain = cfg['pretrain'] # if this is set to true, the dataloader will create its own target input and output based on the provided basemaps
        self.toTensor = ToTensor()
        self.predict = predict # are you using this dataloader to create predictions or to train the model? If you are training, data augmentations such as flipping and rotating will be used. 
        self.split = split # do you want to make a loader for the train, val, or test split, as determined by your splits file at the location described in the configs? 
        self.seed = cfg['Seed'] # theoretically, this should make the random functions do the same thing every time, but I'm not sure that it's working. 

        self.transform = A.Compose([
                A.HorizontalFlip(p=cfg['HorizontalFlip']),
                A.VerticalFlip(p=cfg['VerticalFlip']),
                A.RandomRotate90(p=cfg['RandomRotate90'])
            ]) # this is the transformation that will be used in training (will not be used for predictions)
        
        self.residual = cfg['Residual'] # is your model supposed to predict the difference between the coarse and high resolution image (residual) or simply predict the high resolution image? 
        self.upsample_scale = cfg['upsample_scale'] # factor by which to super-resolve (if coarsening is being done within the dataloader -- this necessarily happens in pretraining)
        
        if self.pretrain == True: # if you are pretraining, find files at these locations
            # data paths
            self.basemap_norm_loc = Path(cfg['pretrain_basemap_norm_loc'])
            self.input_basemap = Path(cfg['pretrain_basemap'])
            self.splits_loc = cfg['pretrain_splits_loc']

            # check that basemaps are in the indicated directory
            if not [file for file in listdir(self.input_basemap) if not file.startswith('.')]:
                raise RuntimeError(f'No input file found in {self.input_basemap}, make sure you put your images there')

        else: # if you are doing regular training and not pretrainig, find files at these locations
            # data paths
            self.coarsen_data = cfg['coarsen_data']
            if not self.coarsen_data:
                self.input_target = Path(cfg['input_target'])
                if not [file for file in listdir(self.input_target) if not file.startswith('.')]:
                    raise RuntimeError(f'No input file found in {self.input_target}, make sure you put your images there')
            self.input_basemap = Path(cfg['input_basemap'])
            self.output_target = Path(cfg['output_target'])
            self.target_norm_loc = Path(cfg['target_norm_loc'])
            self.basemap_norm_loc = Path(cfg['basemap_norm_loc'])
            self.splits_loc = cfg['splits_loc'] # location to split file to indicate which tile belongs to which split

            # check that images are in the given directories
            if not [file for file in listdir(self.input_basemap) if not file.startswith('.')]:
                raise RuntimeError(f'No input file found in {self.input_basemap}, make sure you put your images there')
            if not [file for file in listdir(self.output_target) if not file.startswith('.')]:
                raise RuntimeError(f'No input file found in {self.output_target}, make sure you put your images there')

            # normalization factors for preprocessing
            self.target_norms = pd.read_csv(self.target_norm_loc, delim_whitespace=True)

        # in order to make sure the basemaps aren't large numbers that are difficult to handle, 
        # we normalize them based on the average first and second moments we see on a sample of basemaps. 
        # These moments are stored at self.basemap_norm_loc, and we calculate the averages here. 
        self.basemap_norms = pd.read_csv(self.basemap_norm_loc, delim_whitespace=True).mean()

        # retrieve the information on which files are part of this split
        split_files = [file for file in os.listdir(self.splits_loc) if file.endswith(".csv")] # list out all of the different splits found in the splits_loc directory
        recent_split = sorted(split_files, key=lambda fn:os.path.getctime(os.path.join(self.splits_loc, fn)))[-1] # get most recent split
        split_file = pd.read_csv(os.path.join(self.splits_loc, recent_split))

        # get the ids of tiles in this split
        self.ids = [splitext(file)[0] for file in split_file[split_file['split'] == self.split]['tiles']]

        # keep a copy of the information about the splits since this also has other metadata on the images. 
        self.split_file = split_file # use this later when you get the main landcover of a tile

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self) -> int:
        """Return the number of images in the dataset (specifically this split)."""
        return len(self.ids)

    def preprocess(self, input_basemap_im, input_target_im, output_target_im, target_mean, target_sd):
        """Preprocess the input and output images.

        Args:
            input_basemap_im (Image.Image): Input basemap image.
            input_target_im (Image.Image): Input target image.
            output_target_im (Image.Image): Output target image.
            target_mean (float): Target mean for normalization.
            target_sd (float): Target standard deviation for normalization.

        Returns:
            torch.Tensor: Preprocessed input target image.
            torch.Tensor: Preprocessed output target image.
            torch.Tensor: Preprocessed input image.
            torch.Tensor: Preprocessed output image.
        """
        
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
        ib1, ib2, ib3 = normalize_basemap(input_basemap_im, self.basemap_norms, n_bands=3) # normalize the basemap based on the normalizations set in the class initialization
        output_target = normalize_target(output_target_im, target_mean, target_sd, mean_for_nans=False)

        input = torch.cat([input_target, ib1, ib2, ib3], dim=0)

        return input_target, output_target, input, output
        
    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing the input and output images, labels, and other metadata.
        """
        name = self.ids[idx]
        input_basemap_im = list(self.input_basemap.glob(name + '.tif*'))
        assert len(input_basemap_im) == 1, f'Either no basemap input or multiple basemap inputs found for the ID {name}: {input_basemap_im}'
        if self.pretrain == True:
            input_basemap_im = load(input_basemap_im[0], bands = 3)
            input_basemap_np = np.asarray(input_basemap_im, dtype=np.float32)
            input_basemap_np = shade(input_basemap_np)
            r = input_basemap_np[:,:,0]
            g = input_basemap_np[:,:,1]
            b = input_basemap_np[:,:,2]
            output_target_im, target_mean, target_sd = random_band_arithmetic(r,g,b,seed = self.seed)
            input_target_im = coarsen_image(output_target_im, self.upsample_scale)
        else:
            output_target_im = list(self.output_target.glob(name + '.tif*'))
            assert len(output_target_im) == 1, f'Either no target output or multiple target outputs found for the ID {name}: {output_target_im}'
            output_target_im = load(output_target_im[0], bands = 1)
            if self.coarsen_data:
                input_target_im = coarsen_image(output_target_im, self.upsample_scale)
            else:
                input_target_im = list(self.input_target.glob(name + '.tif*'))
                assert len(input_target_im) == 1, f'Either no target input or multiple target inputs found for the ID {name}: {input_target_im}'
                input_target_im = load(input_target_im[0], bands = 1)
            input_basemap_im = load(input_basemap_im[0], bands = 3)
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
