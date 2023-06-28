'''
    This script will create a file that specifies the training/validation/test split for the data. 
    The split will group tiles by the date and location because nearby flights occured at the same time on a given data collection day. 
    We then split these groups randomly based on the proportions declared at the top of the file.  

    This script also outputs some statistics on the split such that you can qualitatively ensure that there is a relatively even split across train/val/test for 
    1. country/US state 
    2. The time when the tile was collected
    3. The general land cover type

    Also make sure the number of tiles are approximately split according to your chosen proportions since each date and location combination can have very different numbers of tiles. 

    2022 Anna Boser
'''

from logging import raiseExceptions
import os
from numpy import tile
import pandas as pd
import argparse
import yaml
import random
import numpy as np
import re
from datetime import datetime

train_prop = .6
val_prop = .2
test_prop = .2

# load config
parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
args = parser.parse_args()
print(f'Using config "{args.config}"')
cfg = yaml.safe_load(open(args.config, 'r'))


pretrain = cfg['pretrain']
data_root = cfg['data_root']

if pretrain:
    pretrain_basemap = cfg['pretrain_basemap']

    # read in the metadata for each run
    metadata = pd.read_csv(os.path.join(data_root, "metadata", "pretrain_metadata.csv"))
    print(metadata)
    input_files = os.listdir(path = os.path.join(pretrain_basemap))

    RGB_ID = [f.split('_')[0] for f in input_files]

    random_GID = list(set(RGB_ID))
    random.shuffle(random_GID)
    split_dic_train = {gid: 'train' for gid in random_GID[:int(np.floor(train_prop*len(random_GID)))]}
    split_dic_val = {gid: 'val' for gid in random_GID[int(np.floor(train_prop*len(random_GID))):int(np.floor((train_prop + val_prop)*len(random_GID)))]}
    split_dic_test = {gid: 'test' for gid in random_GID[int(np.floor((train_prop + val_prop)*len(random_GID))):]}
    split_dic = split_dic_train | split_dic_val | split_dic_test
    split = [split_dic[gid] for gid in RGB_ID]

    # create a metadata file for each tile with landocver, county, etc info
    zipped = list(zip(input_files, RGB_ID, split))
    tile_df = pd.DataFrame(zipped, columns=["tiles", "ID", "split"])
    tile_df['ID'] = tile_df['ID'].astype(int)
    tile_metadata = pd.merge(tile_df, metadata, on = "ID")

    # save the split

    # create the splits folder in the metadata folder if it doesn't exist already
    output_dir = os.path.join(data_root, "metadata", "pretrain_splits")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get the date-time to name this iteration of the split
    outfile = str(datetime.now()).replace(" ", "_")

    tile_metadata.to_csv(os.path.join(output_dir, outfile + ".csv"))

    # generate and save information about the split

    # generate a file which spits out the configurations as the header, and information about the distribution across the axes discussed above
    def get_split_props():
        splits = list(set(tile_metadata['split']))
        df = pd.DataFrame(columns = splits)
        df.loc[len(df)] = [list((tile_metadata[tile_metadata['split'] == split]['split'])).count(split)/len(tile_metadata) for split in splits]
        return(df)

    def get_props(col):
        types = list(set(tile_metadata[col]))
        splits = list(set(tile_metadata['split']))
        df = pd.DataFrame(columns = [col] + splits)
        for type in types:
            df.loc[len(df)] = [type] + [list((tile_metadata[tile_metadata['split']== split][col])).count(type)/len(tile_metadata[tile_metadata['split'] == split]) for split in splits]
        return(df)

    # occurance of tiles in train, val and test
    split = get_split_props()

    # relative occurance of primarily natural 
    lct = get_props('Main Cover')

    # relative occurance of a country/state
    state = get_props('State')

    # create the splits folder in the metadata folder if it doesn't exist already
    output_dir = os.path.join(data_root, "metadata", "pretrain_splits", "info")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, outfile + ".txt"),"w") as file:
        file.write("Config:\n")
        file.write(str(cfg))
        file.write("\n\n")
        file.write("Number of tiles:\n")
        file.write(str(len(tile_metadata)))
        file.write("\n\n")
        file.write("Number of groups:\n")
        file.write(str(len(set(tile_metadata["ID"]))))
        file.write("\n\n")
        file.write("Split:\n")
        file.write(str(split))
        file.write("\n\n")
        file.write("Land cover:\n")
        file.write(str(lct))
        file.write("\n\n")
        file.write("State/country:\n")
        file.write(str(state))
        file.write("\n\n")

else:
    input_basemap = cfg['input_basemap']
    input_target = cfg['input_target']
    output_target = cfg['output_target']

    # read in the metadata for each run
    metadata = pd.read_csv(os.path.join(data_root, "metadata", "run_metadata.csv"))

    # remove the unusable runs
    metadata = metadata[metadata['Unusable runs'] != 1.0]

    # pool mountainous, desert, and rural into natural and suburban into urban
    def convert(lctype):
        if (lctype == 'Natural' or lctype == 'Urban' or lctype == 'Agricultural'):
            return lctype
        elif (lctype == 'Mountainous' or lctype == 'Desert' or lctype == 'Rural'):
            return 'Natural'
        elif (lctype == 'Suburban'):
            return 'Urban'
        else:
            raise Exception("Unknown landcover type")

    metadata['Main Cover'] = [convert(lct) for lct in metadata['Main Cover']]

    # get the hour of day of image aquisition
    metadata['Hour'] = metadata["Earliest.Time"].apply(lambda x:datetime.strptime(x, '%m/%d/%Y %H:%M')).apply(lambda x: x.hour)

    # list out all of the files in the target and basemap folders
    input_basemap_files = os.listdir(path = os.path.join(input_basemap))
    input_target_files = os.listdir(path = os.path.join(input_target))
    output_target_files = os.listdir(path = os.path.join(output_target))
    files = input_basemap_files + input_target_files + output_target_files

    # make sure there is a matching file in LST and RGB
    unique_files = pd.unique(files)
    assert len(unique_files) == len(files)/3, f'Warning: Not all target files or basemap files have a match'
    # if (len(unique_files) != len(files)/3):
    #     print("Warning: Not all target files or basemap files have a match")

    # extract the run id from the tiles
    Run_ID = [f.split('_')[0] for f in unique_files]
    Group_ID = [re.sub("(?<=[A-z])[0-9]*", "", rid) for rid in Run_ID] # remove the run number after each group name

    # create a random 60/10/30 split grouped by Group_ID, assuming each group is approximately the same size
    random_GID = list(set(Group_ID))
    random.shuffle(random_GID)
    split_dic_train = {gid: 'train' for gid in random_GID[:int(np.floor(train_prop*len(random_GID)))]}
    split_dic_val = {gid: 'val' for gid in random_GID[int(np.floor(train_prop*len(random_GID))):int(np.floor((train_prop + val_prop)*len(random_GID)))]}
    split_dic_test = {gid: 'test' for gid in random_GID[int(np.floor((train_prop + val_prop)*len(random_GID))):]}
    split_dic = split_dic_train | split_dic_val | split_dic_test
    split = [split_dic[gid] for gid in Group_ID]

    # create a metadata file for each tile with landocver, county, etc info
    zipped = list(zip(unique_files, Run_ID, Group_ID, split))
    tile_df = pd.DataFrame(zipped, columns=["tiles", "Run_ID", "Group_ID", "split"])
    tile_metadata = pd.merge(tile_df, metadata, on = "Run_ID")

    # save the split

    # create the splits folder in the metadata folder if it doesn't exist already
    output_dir = os.path.join(data_root, "metadata", "splits")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get the date-time to name this iteration of the split
    outfile = str(datetime.now()).replace(" ", "_")

    tile_metadata.to_csv(os.path.join(output_dir, outfile + ".csv"))

    # generate and save information about the split

    # generate a file which spits out the configurations as the header, and information about the distribution across the axes discussed above
    def get_split_props():
        splits = list(set(tile_metadata['split']))
        df = pd.DataFrame(columns = splits)
        df.loc[len(df)] = [list((tile_metadata[tile_metadata['split'] == split]['split'])).count(split)/len(tile_metadata) for split in splits]
        return(df)

    def get_props(col):
        types = list(set(tile_metadata[col]))
        splits = list(set(tile_metadata['split']))
        df = pd.DataFrame(columns = [col] + splits)
        for type in types:
            df.loc[len(df)] = [type] + [list((tile_metadata[tile_metadata['split']== split][col])).count(type)/len(tile_metadata[tile_metadata['split'] == split]) for split in splits]
        return(df)

    # occurance of tiles in train, val and test
    split = get_split_props()

    # relative occurance of primarily natural 
    lct = get_props('Main Cover')

    # relative occurance of a country/state
    state = get_props('State')

    # relative occurance of hour of day
    hour = get_props('Hour')

    # create the splits folder in the metadata folder if it doesn't exist already
    output_dir = os.path.join(data_root, "metadata", "splits", "info")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, outfile + ".txt"),"w") as file:
        file.write("Config:\n")
        file.write(str(cfg))
        file.write("\n\n")
        file.write("Number of tiles:\n")
        file.write(str(len(tile_metadata)))
        file.write("\n\n")
        file.write("Number of groups:\n")
        file.write(str(len(set(tile_metadata["Group_ID"]))))
        file.write("\n\n")
        file.write("Split:\n")
        file.write(str(split))
        file.write("\n\n")
        file.write("Land cover:\n")
        file.write(str(lct))
        file.write("\n\n")
        file.write("State/country:\n")
        file.write(str(state))
        file.write("\n\n")
        file.write("Time of day:\n")
        file.write(str(hour))
        file.write("\n\n")
