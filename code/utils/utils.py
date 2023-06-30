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


def normalize_basemap(basemap_im, basemap_norms, n_bands=3):

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

def unnormalize_basemap(basemap_im, basemap_norms, n_bands=3):

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

# def randomize(r,g,b, seed = 1234):
#     # Set seed
#     random.seed(seed)
#     # Declare number of r/g/b bands that will be iterated on
#     rgb_num = np.random.randint(2,10)
#     rgb_list = []
#     # Generate list of r/g/b bands to iterate on
#     for i in range(rgb_num):
#         rgb_list.append(np.random.choice(['r','g','b']))
#     op_num = rgb_num - 1
#     ops = {'+':operator.add,
#         '-':operator.sub,
#         '*':operator.mul,
#         '/':operator.truediv}
#     op_list = []
#     counter = 0
#     # Perform correct operation according to proper r/g/b combination
#     for i in range(op_num):
#         # Create list of randomized operations
#         op_list.append(np.random.choice(list(ops.keys())))
#         # Perform first operation between two bands
#         if counter == 0:
#             if (rgb_list[i] == 'r') & (rgb_list[i+1] == 'r'):
#                 rgb_list[i+1] = ops.get(op_list[i])(r,r)
#             elif (rgb_list[i] == 'r') & (rgb_list[i+1] == 'g'):
#                 rgb_list[i+1] = ops.get(op_list[i])(r,g)
#             elif (rgb_list[i] == 'r') & (rgb_list[i+1] == 'b'):
#                 rgb_list[i+1] = ops.get(op_list[i])(r,b)
#             elif (rgb_list[i] == 'g') & (rgb_list[i+1] == 'r'):
#                 rgb_list[i+1] = ops.get(op_list[i])(g,r)
#             elif (rgb_list[i] == 'g') & (rgb_list[i+1] == 'g'):
#                 rgb_list[i+1] = ops.get(op_list[i])(g,g)
#             elif (rgb_list[i] == 'g') & (rgb_list[i+1] == 'b'):
#                 rgb_list[i+1] = ops.get(op_list[i])(g,b)
#             elif (rgb_list[i] == 'b') & (rgb_list[i+1] == 'r'):
#                 rgb_list[i+1] = ops.get(op_list[i])(b,r)
#             elif (rgb_list[i] == 'b') & (rgb_list[i+1] == 'g'):
#                 rgb_list[i+1] = ops.get(op_list[i])(b,g)
#             elif (rgb_list[i] == 'b') & (rgb_list[i+1] == 'b'):
#                 rgb_list[i+1] = ops.get(op_list[i])(b,b)
#             rgb_list[i+1] += 1 # Ensure no division by 0 occurs
#             counter += 1
#         # Perform next operation
#         else:
#             # Normalize data if values start to explode
#             if np.max(np.abs(rgb_list[i])) > 10:
#                 rgb_list[i+1] = ( rgb_list[i] - np.mean(rgb_list[i]) )/ np.std(rgb_list[i]) 
#                 continue
#             elif rgb_list[i+1] == 'r':
#                 rgb_list[i+1] = ops.get(op_list[i])(rgb_list[i],r)
#             elif rgb_list[i+1] == 'g':
#                 rgb_list[i+1] = ops.get(op_list[i])(rgb_list[i],g)
#             elif rgb_list[i+1] == 'b':
#                 rgb_list[i+1] = ops.get(op_list[i])(rgb_list[i],b) 
#     # Store mean and standard deviation of randomized output
#     random_mean = np.mean(rgb_list[-1])
#     random_sd = np.std(rgb_list[-1])
#     # Print mean and standard deviation
#     print('Mean:', random_mean, 'SD:', random_sd, flush = True)
#     # Convert randomized output to Image object
#     randomized_image = Image.fromarray(rgb_list[-1])
#     return randomized_image, random_mean, random_sd # Returns randomized output including its mean and standard deviation


def random_band_arithmetic(red_band, green_band, blue_band, seed = 1234):
    random.seed(seed)
    # Randomly choose the number of arithmetic operations to perform
    num_operations = np.random.randint(1, 4)  # Choose a random integer from 1 to 4
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

# Decides how to open image based on number of specified bands
def load(filename, bands):
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

def coarsen(image, upsample = 8):
    # first, change to 0-1
    print('Image min:', np.nanmin(np.array(image)), flush = True)
    print('Image max:', np.nanmax(np.array(image)), flush = True)
    ds_array = np.array(image)/np.nanmax(np.array(image))
    # Downsample
    downsample_im = skimage.measure.block_reduce(ds_array,
                            (upsample, upsample),
                            np.mean)
    # Resample by a factor of downsample variable with bilinear interpolation
    coarsened_array = Image.fromarray(scipy.ndimage.zoom(downsample_im, upsample, order=1))
    return coarsened_array

def coarsen_image(image, factor):
    # Calculate the new dimensions based on the coarsening factor
    new_width = image.width // factor
    new_height = image.height // factor

    # Resize the image using nearest neighbor interpolation
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)

    # Scale back up to the original size using bilinear interpolation
    coarsened_image = resized_image.resize((image.width, image.height), resample=Image.Resampling.BILINEAR)

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
