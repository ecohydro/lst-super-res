# download quads from planet labs
# adapted from: https://developers.planet.com/apis/orders/basemaps/ 
# author: Mitali Gaidhani

# python has built-in modules - json, os, shutil, random, time, math, yaml
import requests # already satisfied
from requests.auth import HTTPBasicAuth
import json 
import os 
import shutil 
from random import randint 
import time 
import pathlib # not satisfied
from pathlib import Path
import math 
import yaml 

with open('configs/download_planet_labs.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

PLANET_API_KEY = config['api_key'] # setup API key
ORDERS_API_URL = 'https://api.planet.com/compute/ops/orders/v2' # setup Orders API REST URL

session = requests.Session() # setup session
session.auth = (PLANET_API_KEY, "") # authenticate
auth = HTTPBasicAuth(PLANET_API_KEY, '')

data_dir = 'data'
existing_aois = []
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    print(f'Reading contents of directory: {data_dir}')

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            aoi_path = os.path.basename(filepath)
            try:
                aoi_num = aoi_path.split('aoi_', 1)[1]
                aoi_num = aoi_num.split('_', 1)[0]
                print('AOIS in data directory:')
                print(aoi_num)
                existing_aois.append(aoi_num)
            except Exception as e:
                print(f'Error processing file: {filepath}')
                print(f'Error message: {str(e)}')
                continue
else:
    print(f'Directory does not exist: {data_dir}')

# read data from directory with geojson files
directory = config['geojson_path']
data = [] # list for coordinates
filenames = [] # list with filenames
# elements at the same index in the data and filenames list correspond with each other

# print(len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])) # number of files in directory

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if(os.path.isfile(f)):
        # don't read the invisble .DS_Store file
        # .DS_Store is an invisible file on macOS that gets generated when you look into a folder with Finder 
        if(f != directory + '/.DS_Store'): 
            with open(f) as file:
                for item in file:   
                    start = item.find('"coordinates"') + 14 # find coordinates parameter in aoi info
                    end = item.find(',"type"', start) 
                    coords = item[start:end] # read coordinates
                    # print(coords + '\n')
                    data.append(coords)
                    filenames.append(f)

print(len(data)) # extra - hidden DS_Store file 

# make coordinates into nested arrays (they're read from the geojson file as a string)
# for a rectangle polygon - 5 coordinates, adjust accordingly for different polygon
coords_data = []

for i in range(len(data)):
    points = []
    coords = data[i].split('[', 3)[3]
    
    point1 = []
    coordx = coords.split(',', 1)[0]
    coords = coords.split(',', 1)[1]
    point1.append(float(coordx))
    coordy = coords.split(']', 1)[0]
    point1.append(float(coordy))
    coords = coords.split(']', 1)[1]
    points.append(point1)
    
    point2 = []
    coords = coords.split('[', 1)[1]
    coordx = coords.split(',', 1)[0]
    point2.append(float(coordx))
    coords = coords.split(',', 1)[1]
    coordy = coords.split(']', 1)[0]
    point2.append(float(coordy))
    coords = coords.split(']', 1)[1]
    points.append(point2)
    
    point3 = []
    coords = coords.split('[', 1)[1]
    coordx = coords.split(',', 1)[0]
    point3.append(float(coordx))
    coords = coords.split(',', 1)[1]
    coordy = coords.split(']', 1)[0]
    point3.append(float(coordy))
    coords = coords.split(']', 1)[1]
    points.append(point3)
    
    point4 = []
    coords = coords.split('[', 1)[1]
    coordx = coords.split(',', 1)[0]
    point4.append(float(coordx))
    coords = coords.split(',', 1)[1]
    coordy = coords.split(']', 1)[0]
    point4.append(float(coordy))
    coords = coords.split(']', 1)[1]
    points.append(point4)
    
    point5 = []
    coords = coords.split('[', 1)[1]
    coordx = coords.split(',', 1)[0]
    point5.append(float(coordx))
    coords = coords.split(',', 1)[1]
    coordy = coords.split(']', 1)[0]
    point5.append(float(coordy))
    coords = coords.split(']', 1)[1]
    points.append(point5)
    
    geom = []
    geom.append(points)
    # print(geom)
    coords_data.append(geom)


def make_order(coords):
    # select random month and year for the mosaic
    month = str(randint(1, 12)).zfill(2)
    year = str(randint(2016, 2022))
    name = f"global_monthly_{year}_{month}_mosaic"
    
    order_params = {
        "name": "Basemap order with geometry",
        "source_type": "basemaps",
        "products": [
            {
                "mosaic_name": name,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords
                }
            }
        ]
    }
    
    vals = []
    vals.append(order_params)
    vals.append(month)
    vals.append(year)
    
    return vals


def poll_for_success(order_url, auth, num_loops=30):
    count = 0
    while(count < num_loops):
        count += 1
        r = requests.get(order_url, auth=session.auth)
        response = r.json()
        state = response['state']
        print(state)
        end_states = ['success', 'failed', 'partial']
        if state in end_states:
            break
        time.sleep(10)


def download_quad(results, image_number, overwrite=False):
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
#     print('{} items in results'.format(len(results_urls)))
    
    count = 0
    quad_paths = []
    quad_urls = []
    for url, name in zip(results_urls, results_names):
        path = pathlib.Path(os.path.join('data', name))
        
        if(str(path).find("quad.tif") > 0): # only download quads
            count = count + 1
            quad_paths.append(path)
            quad_urls.append(url)

    quad_index = math.ceil(count / 2)
    quad_to_download = quad_paths[quad_index - 1]
    quad_url = quad_urls[quad_index - 1]
    
    if overwrite or not quad_to_download.exists():
        print('downloading {} to {}'.format(name, quad_to_download))
        r = requests.get(quad_url, allow_redirects=True)
        quad_to_download.parent.mkdir(parents=True, exist_ok=True)
        
        open(quad_to_download, 'wb').write(r.content)
    else:
        print('{} already exists, skipping {}'.format(quad_to_download, name))
    
    return quad_to_download

def planet_search(coords, image_num):
    order = make_order(coords)
    order_params = order[0]
    month = order[1]
    year = order[2]
    
    headers = {"content-type": "application/json"}
    paramRes = requests.post(ORDERS_API_URL, data=json.dumps(order_params), auth=auth, headers=headers)
    
    order_url = ORDERS_API_URL + '/' + paramRes.json()['id']
    # while True:
    #     try:
            # order_url = ORDERS_API_URL + '/' + paramRes.json()['id']
            # break
        # except KeyError:
        #     print("no basemap quads were found, trying different order parameters")
        #     order = make_order(coords)
        #     order_params = order[0]
        #     month = order[1]
        #     year = order[2]
        #     print('order_params = ' + str(order_params))

        #     headers = {"content-type": "application/json"}
        #     paramRes = requests.post(ORDERS_API_URL, data=json.dumps(order_params), auth=auth, headers=headers)
            
    poll_for_success(order_url, auth)
    r = requests.get(order_url, auth=session.auth)
    response = r.json()
    
    results = response['_links']['results']
    old_path = download_quad(results, image_num)
    quad_id = str(os.path.basename(old_path)).split('_', 1)[0]
    new_path = str(os.path.split(old_path)[0]) + '/aoi_' + str(image_num) + '_quad_' + quad_id + '_year_'+ str(year) + '_month_' + str(month) + '.tif'

    os.rename(old_path, new_path)


count = 0
# add image number in saved quad data
for i in range(len(coords_data)):
    print('Number of files to download: ' + filenames[count])
    file_num = filenames[count]
    file_num = os.path.basename(file_num)
    file_num = file_num.split('.', 1)[0]

    if file_num not in existing_aois:
        print('Ordering AOI ' + file_num)
        while True:
            try:
                planet_search(coords_data[count], file_num)
                break
            except Exception as e:
                print('An error occured:', str(e))
                continue
    else:
        print('Skipping AOI ' + file_num)
    
    count = count + 1


print('Copying downloaded quads to ' + config['quad_download_path'])
new_dir = Path(config['quad_download_path'])
new_dir.mkdir(parents=True, exist_ok=True)

tif_paths = list(Path("data").rglob("*.tif"))

for path in tif_paths:
    path_str = str(path)
    filename = os.path.basename(path_str)
    new_path = new_dir / filename
    shutil.move(path, new_path)

# delete original download directory
print('Deleting temporary directory')
shutil.rmtree('data')
