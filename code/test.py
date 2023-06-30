from PIL import Image
import os
import numpy as np
import tifffile
import random

# def get_summary_statistics(arr):
#     # Calculate summary statistics
#     minimum = np.min(arr)
#     maximum = np.max(arr)
#     mean = np.mean(arr)
#     median = np.median(arr)
#     std_deviation = np.std(arr)
#     variance = np.var(arr)

#     # Return summary statistics as a dictionary
#     summary_stats = {
#         'Minimum': minimum,
#         'Maximum': maximum,
#         'Mean': mean,
#         'Median': median,
#         'Standard Deviation': std_deviation,
#         'Variance': variance
#     }

#     return summary_stats



# filename = '/home/waves/projects/lst-super-res/For_pretrain/tiles/38_09_2022_1999E-1561N_tile_4,2.5.tif'
# nothing = '/home/waves/projects/lst-super-res/For_pretrain/tiles/2_03_2023_1238E-1092N_tile_0,0.tif'

# #print(os.stat(filename).st_size)
# img = tifffile.imread(filename)
# print(img.shape)
# print(type(img))

# img_data = img[:,:,:3]
# stats = get_summary_statistics(img_data)
# print(stats)
# print(img_data.shape)
# #Image.fromarray((img_data * 1).astype(np.uint8)).convert('RGB')


# def load(filename, bands):
#     try:
#         if bands == 1:
#             return Image.open(filename)
#         elif bands == 3:
#             return Image.open(filename).convert('RGB')
#         else: 
#             raise ValueError('Image must be one or three bands')
#     except Exception as e:
#         try:
#             image_data = tifffile.imread(filename)
#             if bands == 1:
#                 # Take the first band if multiple are present
#                 if len(image_data.shape) > 2:
#                     image_data = image_data[:, :, 0]
#                 return Image.fromarray((image_data * 1).astype(np.uint8))
#             elif bands == 3:
#                 # Take the first three bands if available
#                 if len(image_data.shape) > 2 and image_data.shape[2] >= 3:
#                     image_data = image_data[:, :, :3]
#                 return Image.fromarray((image_data * 1).astype(np.uint8)).convert('RGB')
#             else:
#                 raise ValueError('Image must be one or three bands')
#         except Exception as e:
#             raise ValueError('Failed to load the image: {}'.format(e))

# image = load(filename, 3)


# def random_band_arithmetic(red_band, green_band, blue_band):
#     # Randomly choose the number of arithmetic operations to perform
#     num_operations = random.randint(1, 4)  # Choose a random integer from 1 to 4

#     # Create a list of bands to choose from
#     bands = [red_band, green_band, blue_band]

#     # Shuffle the bands list
#     random.shuffle(bands)

#     # Perform random arithmetic operations
#     result = bands[0].copy()
#     for i in range(1, num_operations):
#         band = bands[i]  # Select the band from the shuffled list

#         operation = random.choice(['+', '-', '*', '/'])  # Randomly select the arithmetic operation

#         if operation == '+':
#             result += band
#         elif operation == '-':
#             result -= band
#         elif operation == '*':
#             result *= band
#         elif operation == '/':
#             # Ignore division by zero and assign a small value instead
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 result = np.divide(result, band, out=np.zeros_like(result), where=band != 0)

#     return result

# red_band = np.random.rand(10, 10)  # Random red band array
# green_band = np.random.rand(10, 10)  # Random green band array
# blue_band = np.random.rand(10, 10)  # Random blue band array

# result = random_band_arithmetic(red_band, green_band, blue_band)
# print(result)



from utils.utils import coarsen

image = Image.open('/home/waves/projects/lst-super-res/For_CNN/inputs/RGB_672_10/20160112AlisoCanyonNHighCA1_698-2464_tile0,0.5.tif')

def coarsen_image(image, factor):
    # Calculate the new dimensions based on the coarsening factor
    new_width = image.width // factor
    new_height = image.height // factor

    # Resize the image using nearest neighbor interpolation
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)

    # Scale back up to the original size using bilinear interpolation
    coarsened_image = resized_image.resize((image.width, image.height), resample=Image.Resampling.BILINEAR)

    return coarsened_image

# Coarsen the image with a factor of 2 (halving the dimensions)
coarsened_image = coarsen_image(image, factor=8)

# Save the coarsened image
coarsened_image.save('coarsened_image.tif')