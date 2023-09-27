# Individually plot out RGB, R/G/B, GT LST bands

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from code.utils.utils import load

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('exit-ramp.jpg')
plt.imshow(image)
plt.show()

# Step 1: Load the TIFF file and split into bands
tiff_file_path = '/home/waves/projects/lst-super-res/data/For_CNN/inputs/RGB_672_10/20210524LosAngelesCA5_tile5,1.tif'
lst_file_path = '/home/waves/projects/lst-super-res/data/For_CNN/inputs/LST_672_70_10/20210524LosAngelesCA5_tile5,1.tif'

lst_data = np.array(load(lst_file_path, 1))
plt.imshow(lst_data)
plt.axis('off')
png_filename = 'RGB_plot/test_LST.png'
plt.savefig(png_filename, bbox_inches='tight', pad_inches=0, dpi=300)

gt_file_path = '/home/waves/projects/lst-super-res/data/For_CNN/outputs/LST_672_10/20210524LosAngelesCA5_tile5,1.tif'

gt_data = np.array(load(gt_file_path, 1))
plt.imshow(gt_data)
plt.axis('off')
png_filename = 'RGB_plot/test_GT.png'
plt.savefig(png_filename, bbox_inches='tight', pad_inches=0, dpi=300)

tiff_data = load(tiff_file_path, 3)

plt.imshow(tiff_data)
plt.axis('off')
png_filename = 'RGB_plot/test_RGB.png'
plt.savefig(png_filename, bbox_inches='tight', pad_inches=0, dpi=300)

r_band, g_band, b_band = tiff_data.split()

# Step 2: Convert bands to numpy arrays
r_array = np.array(r_band)
g_array = np.array(g_band)
b_array = np.array(b_band)

# Step 3: Plot and save each band
bands = [('Red', r_array), ('Green', g_array), ('Blue', b_array)]

for band_name, band_array in bands:
    # Plot
    if band_name == 'Red':
        plt.imshow(band_array, cmap='Reds')  # For single-channel images like bands
        plt.axis('off')
    
    if band_name == 'Green':
        plt.imshow(band_array, cmap='Greens')  # For single-channel images like bands
        plt.axis('off')

    if band_name == 'Blue':
        plt.imshow(band_array, cmap='Blues')  # For single-channel images like bands
        plt.axis('off')
    
    # Save as PNG
    png_filename = f'RGB_plot/test_{band_name.lower()}_band.png'
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    
    # Clear plot for the next iteration
    plt.clf()

# Close the TIFF file
tiff_data.close()