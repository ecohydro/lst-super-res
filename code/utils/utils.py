'''
    This script contains miscellaneous util functions that are declared in other .py files.

    2022 Anna Boser
'''

import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import operator
from PIL import Image 
import skimage
import scipy.ndimage
from scipy import stats
import tifffile
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skimage import img_as_float
from skimage.metrics import structural_similarity
from torch import Tensor
from typing import Tuple

def normalize_target(target_im: Tensor, target_mean: float, target_sd: float, mean_for_nans: bool = True) -> Tensor:
    """
    Normalize the target image tensor based on the provided mean and standard deviation.

    Args:
        target_im (Tensor): The target image tensor to normalize.
        target_mean (float): The mean value used for normalization.
        target_sd (float): The standard deviation value used for normalization.
        mean_for_nans (bool, optional): Flag indicating whether to replace NaN values with zero. 
                                        Defaults to True.

    Returns:
        Tensor: The normalized target image tensor.
    """

    target_im[target_im<=-3.4e+30] = float('nan')      # -3.4e+38 to NaN

    # normalize
    target_im = (target_im - target_mean)/target_sd

    if mean_for_nans==True:
        target_im[torch.isnan(target_im)] = 0
    
    return target_im

def unnormalize_target(target_im: Tensor, target_mean: float, target_sd: float) -> Tensor:
    """
    Unnormalize the target image tensor based on the provided mean and standard deviation.

    Args:
        target_im (Tensor): The normalized target image tensor to unnormalize.
        target_mean (float): The mean value used for normalization.
        target_sd (float): The standard deviation value used for normalization.

    Returns:
        Tensor: The unnormalized target image tensor.
    """
    # Unnormalize
    target_im = (target_im*target_sd) + target_mean
    
    return target_im


def normalize_basemap(basemap_im: Tensor, basemap_norms: dict, n_bands: int = 3):
    """
    Normalize the basemap image tensor based on the provided mean and standard deviation.

    Args:
        basemap_im (Tensor): The basemap image tensor to normalize.
        basemap_norms (dict): A dictionary containing the mean and standard deviation values for each band.
        n_bands (int, optional): The number of bands in the basemap. Defaults to 3.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The normalized image tensors for each band.
    """

    assert n_bands == 3, \
        'Currently only basemaps with three bands are supported.'

    # retrieve the mean and sds from the basemap
    basemap_mean1 = basemap_norms['mean1'] 
    basemap_mean2 = basemap_norms['mean2']
    basemap_mean3 = basemap_norms['mean3']
    basemap_sd1 = basemap_norms['sd1']
    basemap_sd2 = basemap_norms['sd2']
    basemap_sd3 = basemap_norms['sd3']

    # normalize
    input_1 = (basemap_im[[0]] - basemap_mean1)/basemap_sd1
    input_2 = (basemap_im[[1]] - basemap_mean2)/basemap_sd2
    input_3 = (basemap_im[[2]] - basemap_mean3)/basemap_sd3

    return input_1, input_2, input_3

def unnormalize_basemap(basemap_im: Tensor, basemap_norms: dict, n_bands: int = 3) -> Tensor:
    """
    Unnormalize the basemap image tensor based on the provided mean and standard deviation.

    Args:
        basemap_im (Tensor): The basemap image tensor to unnormalize.
        basemap_norms (dict): A dictionary containing the mean and standard deviation values for each band.
        n_bands (int, optional): The number of bands in the basemap. Defaults to 3.

    Returns:
        Tensor: The unnormalized basemap image tensor.
    """

    assert n_bands == 3, \
        'Currently only basemaps with three bands are supported.'

    # retrieve the mean and sds from the basemap
    basemap_mean1 = basemap_norms['mean1'] 
    basemap_mean2 = basemap_norms['mean2']
    basemap_mean3 = basemap_norms['mean3']
    basemap_sd1 = basemap_norms['sd1']
    basemap_sd2 = basemap_norms['sd2']
    basemap_sd3 = basemap_norms['sd3']

    # normalize
    input_1 = (basemap_im[[0]]*basemap_sd1) + basemap_mean1
    input_2 = (basemap_im[[1]]*basemap_sd2) + basemap_mean2
    input_3 = (basemap_im[[2]]*basemap_sd3) + basemap_mean3

    return (torch.cat([input_1, input_2, input_3], dim=0)).int()

def unnormalize_image(normalized_image, basemap_norms, target_norms, n_bands = 3):

    assert n_bands == 3, \
        'Currently only basemaps with three bands are supported.'
    
    # retrieve the mean and sds from the basemap
    target_mean = target_norms['mean']
    target_sd = target_norms['sd']

    # normalize
    target_im = (normalized_image[[0]]*target_sd) + target_mean

    # retrieve the mean and sds from the basemap
    basemap_mean1 = basemap_norms['mean1'] 
    basemap_mean2 = basemap_norms['mean2']
    basemap_mean3 = basemap_norms['mean3']
    basemap_sd1 = basemap_norms['sd1']
    basemap_sd2 = basemap_norms['sd2']
    basemap_sd3 = basemap_norms['sd3']

    # normalize
    input_1 = (normalized_image[[1]]*basemap_sd1) + basemap_mean1
    input_2 = (normalized_image[[2]]*basemap_sd2) + basemap_mean2
    input_3 = (normalized_image[[3]]*basemap_sd3) + basemap_mean3

    return (torch.cat([target_im,input_1,input_2,input_3], dim=0)).int()


def random_band_arithmetic(red_band: np.ndarray, green_band: np.ndarray, blue_band: np.ndarray,
                           seed: int = 1234) -> Tuple[Image.Image, float, float]:
    """
    Perform random arithmetic operations on the input bands and return the result along with its mean and standard deviation.

    Args:
        red_band (np.ndarray): The red band.
        green_band (np.ndarray): The green band.
        blue_band (np.ndarray): The blue band.
        seed (int, optional): The seed value for random number generation. Defaults to 1234.

    Returns:
        Tuple[Image.Image, float, float]: A tuple containing the resulting image, mean, and standard deviation.
    """
    random.seed(seed)
    # Randomly choose the number of arithmetic operations to perform
    num_operations = np.random.randint(1, 4)  # Choose a random integer from 1 to 3
    num_copies = np.random.randint(1, 4)

    # Create a list of bands to choose from
    bands = [red_band, green_band, blue_band]

    # Shuffle the bands list
    np.random.shuffle(bands)
    copied_bands = bands.copy()
    for _ in range(num_copies - 1):
        copied_bands.extend(bands)  # Extend the copied list with the original list

    # Perform random arithmetic operations
    result = bands[0].copy()
    for i in range(1, num_operations):
        band = copied_bands[i]  # Select the band from the shuffled list

        operation = np.random.choice(['+', '-', '*', '/'])  # Randomly select the arithmetic operation

        if operation == '+':
            result += band
        elif operation == '-':
            result -= band
        elif operation == '*':
            result *= band
        elif operation == '/':
            # Ignore division by zero and assign a small value instead
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(result, band, out=np.zeros_like(result), where=band != 0)
    random_mean = result.mean()
    random_sd = result.std()
    print('Mean:', random_mean, 'SD:', random_sd, flush = True)
    result = Image.fromarray(result) # Convert np array to image object
    return result, random_mean, random_sd

def load(filename: str, bands: int) -> Image.Image:
    """
    Load an image from a file and return it as a PIL Image object. 
    Try a couple of ways if necessary. 

    Args:
        filename (str): The path to the image file.
        bands (int): The number of bands in the image.

    Returns:
        Image.Image: The loaded image as a PIL Image object.

    Raises:
        ValueError: If the image has an invalid number of bands or if loading the image fails.
    """
    try:
        if bands == 1:
            return Image.open(filename)
        elif bands == 3:
            return Image.open(filename).convert('RGB')
        else: 
            raise ValueError('Image must be one or three bands')
    except Exception as e:
        try:
            image_data = tifffile.imread(filename)
            if bands == 1:
                # Take the first band if multiple are present
                if len(image_data.shape) > 2:
                    image_data = image_data[:, :, 0]
                return Image.fromarray((image_data * 1).astype(np.uint8))
            elif bands == 3:
                # Take the first three bands if available
                if len(image_data.shape) > 2 and image_data.shape[2] >= 3:
                    image_data = image_data[:, :, :3]
                return Image.fromarray((image_data * 1).astype(np.uint8)).convert('RGB')
            else:
                raise ValueError('Image must be one or three bands')
        except Exception as e:
            raise ValueError('Failed to load the image: {}'.format(e))

def coarsen_image(image, factor):
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Create a copy of the image array
    coarsened_array = np.copy(image_array)

    # Find NaN values in the image array
    nan_indices = np.isnan(image_array)

    # Calculate the new dimensions based on the coarsening factor
    new_width = image.width // factor
    new_height = image.height // factor

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
            average_value = np.mean(valid_pixels) if valid_pixels.size > 0 else 0

            # Assign the average value to the corresponding pixels in the coarsened array
            coarsened_array[start_y:end_y, start_x:end_x][nan_indices[start_y:end_y, start_x:end_x]] = average_value
    # Convert the coarsened array back to an image
    coarsened_image = Image.fromarray(coarsened_array)

    # Scale back up to the original size using nearest neighbor interpolation
    coarsened_image = coarsened_image.resize((image.width, image.height), resample=Image.Resampling.NEAREST)
    return coarsened_image


def get_r2(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(r2_score(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_mse(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(mean_squared_error(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_mae(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(mean_absolute_error(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_ssim(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    img_1 = img_1[~np.isnan(img_1)]
    img2 = img2[~np.isnan(img2)]
    result = structural_similarity(img_1, img2, data_range = img_1.max()-img_1.min()) 
    return round(result, 2)
