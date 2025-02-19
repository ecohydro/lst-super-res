# lst-super-res
This repository constitutes a U-Net model whose goal is to increase the resolution of a low resolution image (1 band) with the help with a seperate high resolution input (3 bands). Concretely, this model was developed to increase the resolution of Land Surface Temperate (LST) images from 70m to 10m with the help of a 10m RGB basemap. The code for our U-Net was adapted from https://github.com/milesial/Pytorch-UNet. 

This code is highly flexible and, as with the U-Net implementation we borrow our basic structure from, takes any resonably sized image (try ~300-2000 pixels on each side). There are two inputs into the model: a *basemap* (in our case RGB), which should be at the resolution of the desired output, and a *coarse target* (in our case LST) which should be at the desired resolution of your original image you are hoping to increase the resolution of, but resized to the same resolution as the basemap. The output which the model will be trained on should be the same size and resolution as the basemap input. 

Because high resolution training data of the target of choice is not always very available, the model also includes a pre-training feature wherein the model can create artificial data from basemap data and the model learns to highten the resolution of this artificial data, at which these weights can be transfered to the task using real target data. We include code to download and process RGB basemaps from PlanetLabs with some information on different land covers, though one must provide their own API key. 

Finally, a pixel-level Random Forest regressor is also available as a benchmark for performance on our various evaluation metrics.

## Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

```bash
conda create -n lst-super-res python=3.9 -y
conda activate lst-super-res
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
```

_Note:_ depending on your setup you might have to install a different version of PyTorch (e.g., compiled against CUDA). See https://pytorch.org/get-started/locally/

3. Add data

The processed dataset for this particular project is currently not publicly available. However, one should manually add inputs and outputs and specify their paths in a configs/*.yaml file. Speficially, you will need to add paths to three folders (or only two if `coarsen_data` is set to True):

- `input_basemap`: This folder is constituted of 8-bit, 3-band images of your basemap of choice (in our case RGB), all the same size and resolution (e.g. 672x672 pixels at 10m resolution)
- `output_target`: This folder is constituted of your labels: single band floating point images of your target (in our case LST) at the desired improved resolution (e.g. 10m in our case.) 
- `input_target` (optional): This folder is constituted of single band floating point images of your target (in our case LST) at a coarse resolution (e.g. 70m) but resampled to the same size and resolution as the basemap and the desired output. This folder is optional because you can also set `coarsen_data` to `True` in the configs file and have the utils function **coarsen_image** coarsen the data for you on the fly. However, saving the data seperately will save on computational power if you will be training for many epochs on the same coarsened resolution. If opting to use the built in `coarsen_data` option, make sure to also specify the `upsample_scale` if choosing this option. 

_Note:_ Corresponding images from matching scenes should be named the same between folders, or you will get an error. 

The location of some metadata must also be included in your configs file:
- `splits_loc`: This the location where you would like the file that determines how your dataset is to be split to be stored. This file is created in step 4: 'Split data', and is a csv with the name of an image and whether it belongs to the "train", "val" or "test" set. The most recent file in this folder are used as your split.
- `target_norm_loc`: This is a space delimited file that includes mean and sd columns with entries for all of your target input images (in our case, LST). The average across these are taken to normalize the inputs. [This is being updated and likely be obsolete since mean and sd of. thetarget norm will be calcualted in the dataloader]. 
- `basemap_norm_loc`: This is a space delimited file that includes mean (mean1, mean2, mean3) and sd (sd1, sd2, sd3) columns with entries for all of your input basemap images (in our case, RGB). The average across these are taken to normalize the inputs. This file is to be used both during pre-training and regular training of the model. 

- The metadata on the runs which includes information on their land cover type, `runs_metadata.csv` should be stored in a folder named "metadata" which is within your `data_root` as specified in your configs file. 

_Note:_ It is OK to have NA values in the input and output target, but not in your basemaps. There is built-in functionality to ignore areas where there is no information for the target: input NAs are set to 0 and output NAs are ignored when calculating the loss.

If you are interested in pre-training your model, you will need to provide the following paths and other configurations must be specified in the configs/*.yaml file (or instead, if you are only doing pretraining):

- `pretrain`: This is a boolean value that when set to `TRUE`, tells the model to perform pre-training.

- `pretrain_input_basemap`: This folder is constituted of 8-bit, 3-band images of your basemap of choice (in our case RGB), all the same size and resolution (e.g. 672x672 pixels at 10m resolution)

- `pretrain_splits_loc`: This the location of your file that determines how your dataset is to be split. It should contain a csv with the name of an image and whether it belongs to the "train", "val" or "test" set. The most recent file in this folder are used as your split.

- `pretrain_basemap_norm_loc`: This is a space delimited file that includes mean (mean1, mean2, mean3) and sd (sd1, sd2, sd3) columns with entries for all of your pre-training input basemap images (in our case, RGB). The average across these are taken to normalize the inputs.

The metadata on these high resolution RGB images which includes information on their land cover type, `pretrain_metadata.csv` should be stored in a folder named "metadata" which is within your `data_root` as specified in your configs file.

4. Split data

To create a data split, you will need to have your data available in the location specified in your configs file and your metadata file, `data_root/runs_metadata.csv`, which will be used to ensure your splits are even across different variables. 

_Note:_ If `pretrain` is set to `True` in your configuration file, the metadata information should be stored as a CSV file under `data_root/pretrain_metadata.csv`. The output folder will be `data_root/metadata/pretrain_splits`.

```bash
python3 code/split.py --config configs/base.yaml
```

This will create `data_root/metadata/splits`, a folder containing a CSV file that indicates which observation belongs to each split and a .txt file that provides additional information regarding the split. Note that `data_root` is declared in your specified configuration file.

Make sure to check if you consider your split to be adequately distributed across the variables of interest specified in your metadata file. 

## Reproduce results

For each desired experiment, create a `*.yaml` configuration file that has the same variable declarations as in the provided example configuration `configs/base.yaml`. Ensure that when running each file, you point to the correct configuration file by using the `--config` argument. Note: Default configuration file is set to `configs/base.yaml`.

1. Train

```bash
python code/train.py --config configs/base.yaml
```

This will create a trained model which is saved at each epoch in the checkpoints folder, `experiment_dir`. This folder contains model checkpoints, a copy of the configuration file used, and a copy of split info used during training. The path to this directory is declared in your configuration file. *Note:* **Ensure `experiment_dir` is unique for each experiment as to not overwrite previous experiments.**

During training, weights and biases (wandb) is used to automatically generate visualizations of the training data and plot out the loss (MSE) of the training and validation sets. Wandb logs are generated and saved in the folder `code/wandb`. 

If you would like to load in a previously trained model for further training, use `--load` followed by the path to the model (must be a `.pth` file). Also, specifiy how many epochs the model was trained for in the configuration file under `epochs_done`.

2. Predictions and validation

Generate predictions and evaluation metrics. Predictions and associated metrics will be saved in the predictions folder of your experiment. If predictions are desired for another split, you can also specify 'test' or 'train'. If `--visualize` is set to `True`, this file will also generate visualization plots for each prediction. Note that you can generate predictions and metrics on data other than what you trained on: simply specify new paths in the configs file or change the pretraining status as you desire before this step. You can also validate one of the models saved at different epochs during the training process: simply specify `--model_epoch` followed by the epoch number. 

```bash
python code/predict.py --config configs/base.yaml --split train --visualize True
python code/predict.py --config configs/base.yaml --split val --visualize True
```
This will create the following folders and files:

`experiment_dir/predictions`: A folder containing all predicted target images separated by split. 

`experiment_dir/prediction_metrics`: A folder containing a CSV file that includes evaluation metrics (R2, SSIM, MSE) for each prediction separated by split

`experiment_dir/prediction_plots`: A folder containing PNG files that includes the basemap image, coarsened target image, predicted target image, and ground truth image for each prediction separated by split. Also shows image name, landcover type, prediction metrics and coarsened input metrics. Note: This folder is created only if `--visualize` is set to `True`.

*Note:* If `pretrain` is set to `True` in your configuration file, the predictions will be based on your _pretraining data_ and will be stored in `experiment_dir/pretrain_predictions`, and prediction metrics and plots will be stored in `experiment_dir/pretrain_prediction_metrics` and `experiment_dir/pretrain_prediction_plots` respectively.

3. Test/inference

Once you are ready to test your model on your held out test set, run the following: 

```bash
python code/predict.py --config configs/base.yaml --split test
```

4. (Optional) Convert pretraining model to training model

Repeat Steps 1-3 once again, but during Step 1, use `--load` followed by the path to the pretrained model (must be a `.pth` file).

```bash
python code/train.py --config configs/base.yaml --load path/to/pretrained/model.pth
```

## Random Forest Regressor Model

The random forest regressor model, which represents the state-of-the-art approach prior to the U-Net model we implement for enhancing the resolution of land surface temperature images, employs a statistical pixel-based technique. In order to evaluate its performance against our custom U-Net model, we employ the random forest regressor.

```bash
python code/RF.py --config configs/base.yaml --split train
python code/RF.py --config configs/base.yaml --split val
python code/RF.py --config configs/base.yaml --split test
```

This will produce `/RF/results.csv`, a CSV file that includes the file name, landcover type, as well as the R2 and RMSE values.


## File Table

| File Name and Location  | Description |
| ------------- | ------------- |
| `code/dataset_class.py`  | This script creates the dataset class to read and process data to feed into the dataloader. The Dataset class is called for both model training and predicting in `code/train.py` and `code/predict.py` respectively.|
| `code/evaluate.py`  | This script evaluates the validation score for each epoch during training. It is declared in `code/train.py`.|
| `code/predict_vis.py`  | This script contains functions for all evaluation metrics and creating PNG images for each prediction. These predictions are evaluated using MSE, SSIM, and R2_score metrics.|
| `code/predict.py`  | This script performs predictions using the trained UNet model and computes evaluation metrics (R2, RMSE, SSIM scores) on desired split. Optionally, can generate visualization plots for each prediction.  |
| `code/RF.py`  | This script enhances coarsened LST images using a Random Forest regressor.  |
| `code/split.py`  | This script will create a file that specifies the training/validation/test split for the data.  |
| `code/train.py`  | This script trains a U-Net model given 3 channel basemap images and 1 channel coarsened target image to predict a 1 channel high resolution target image.  |
| `utils/utils.py`  | This script contains miscellaneous util functions that are declared in other .py files.  |
| `unet/unet_model.py`  | This script contains the full assembly of the U-Net parts to form the complete network  |
| `unet/unet_parts.py`  | This script conatins class definitions of each part of the U-Net model  |



#### For Anna

Here is a quick overview of the some of the changes that were implemented during Summer 2023:

- A couple new files were made in the `code` folder: `plot_test.py` which plots out the RGB image overlapped with the high res LST data in order to see where tiles are off-centered (all images will be found in `plots` folder) and `edges.py` which creates ane edge mask for the RGB image which can then be used for model training (all images will be found in `edges` folder).

- Attempted to make a Docker image for this project (file is called `Docker2` and is still a WIP). Was following this guide: https://docs.docker.com/get-started/02_our_app/

- For using splits, please use `metadata/ryan_lst_#_train_splits` where # refers to the number of flights used for model training (All of these csv splits are in a shared Google Drive folder under LST super-res https://docs.google.com/spreadsheets/d/1SY5CmhW-3CCW6qzMSXIpewnFGe8K8yfhrpBUimmZZBE/edit?usp=sharing). When using all data, please use `metadata/ryan_lst_edited_val_splits` as it has removed unnecessary tiles. For reference, data is cleaned by:
    - Removing tiles in which 90% or more of the data is ocean
    - Shakiness, blurriness, off-centeredness (mainly from LST images in urban settings)
    - Any objects that may obstruct the view of the RGB image (mainly just clouds)

- `shade()` util function added to `utils` folder which creates a random polygon object from 3 to 10 vertices and adds that polygon as a shadow to the randomized band for pretraining.

- `code/RGB_plot.py` is a simple visualization file used for showing RGB, R, G, B, and Ground Truth LST bands of a single tile and saves them to a folder called `RGB_plot`

- Additional metrics are now generated by `predict.py` including `prediction_metrics_#/prediction_group_metrics_mean.csv` which has metrics grouped by flights and `prediction_metrics_#/prediction_lc_metrics_mean.csv` which has metrics grouped by land cover type.

- Updated `RF.py` to run properly after changes implemented to other files.

- Observations made:

    - Reducing number of flights actually did not negatively affect predictions that much until minimized to the lowest amount of flights. Also, model training seems to be optimized by around its 25th epoch. Example metrics of different flights after 25 epochs (Note: These are base LST models w/o pre-training): R2 Coarse (BASELINE): 0.7117, 1 Flight: 0.695, 2 Flights: 0.734, 3 Flights: 0.721, 4 Flights: 0.735, 5 Flights: 0.752, 10 Flights: 0.746, 15 Flights: 0.764, All flights: 0.771. 5 flights had a total of 186 tiles to train on, 10 had 459 tiles, and 15 had 723, and all 20 flights are roughly 1000ish tiles.

    - Attempted to diagnose why pretraining method was not as effective as intended: `plot_test.py` was used to check mismatching in overlapping pixels in RGB vs LST image. Noticed that although most images were okay, some were mismatched and for certain images such as the Santa Barbara runs, the coasts would not align due to changes in tide. We also noticed some images were interpreting the RGB images too literally, mainly could be seen in coastal images where waves from RGB image would show up in LST prediction, though there was not that much of a decrease in prediction values. If given more time, we would want to possibly increase the complexity in the randomization function.

    - In terms of transfer learning, I used `experiments/ryan_pret_512_train/checkpoints/checkpoint_epoch75.pth` for a majority of transfer models as it had the best results at the time (R2 Pred: 0.733 vs R2 Coarse: 0.716), although `/home/waves/projects/lst-super-res/experiments/ryan_pret_512_shade/checkpoints/checkpoint_epoch20.pth` is now marginally the best (R2 Pred: 0.736 vs R2 Coarse: 0.716) as transfer models with randomized shading has better metrics for lower # of training flights. Ex: 5 flights @ 25 epochs (R2 transfer: .747; R2 transfer with shading in pretraining: .762; R2 Coarse: .712), 5 flights @ 25 epochs (R2 transfer: .741; R2 transfer with shading in pretraining: .755; R2 Coarse: .712).

    - Attempted to change U-net for `n_class=5` as input, but ran into errors when initializing model (mainly because I was adding new config parameters such as pathing to edge .png files, etc) and initializing new variables in `train.py` and `dataset_class.py`. Definitely able to change the model to have a 5 channel input, just ran out of time to implement it.