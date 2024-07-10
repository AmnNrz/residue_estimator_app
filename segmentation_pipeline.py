# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: res_app
#     language: python
#     name: python3
# ---

# +
import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib as mpl
from pandas import read_csv, read_excel, DataFrame
import os

from skimage.feature import local_binary_pattern as LBP
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle
import logging
import glob
from collections import defaultdict
from utils_segmentation import (
    get_features,
    p3,
    p0,
    p00,
    n_components,
    plots,
    cornfusion,
)


logging.basicConfig(level=logging.INFO)

# +
##### Read an rgb image with its labels #####
path_to_data = ("/mnt/C250892050891BF3/Projects/residue_estimator/images/")

### Function to find images
def find_jpg_images(root_dir):
    jpg_image = []
    for dir_name,a,b in os.walk(root_dir):
        search_pattern = os.path.join(dir_name, '*.jpg')
        for filename in glob.glob(search_pattern):
            jpg_image.append(filename)
    return jpg_image    

# Function to find labels
def find_tif_labels(root_dir):
    tif_labels = []
    for dir_name, _, _ in os.walk(root_dir):
        search_pattern = os.path.join(dir_name, '*.tif')
        for filename in glob.glob(search_pattern):
            tif_labels.append(filename)
    return tif_labels

# Check if labels exist for each image
path_to_original = path_to_data + "original/"
original_paths = find_jpg_images(path_to_original)

path_to_labels = path_to_data + "label/"
label_paths = find_tif_labels(path_to_labels)

# Extract image numbers from file paths
def extract_image_number(file_path):
    base_name = os.path.basename(file_path)
    return base_name.split('_')[1].split('.')[0]

# Group labels by image number
label_dict = defaultdict(set)
for label in label_paths:
    image_number = extract_image_number(label)
    label_dict[image_number].add(label)

# Find valid image numbers that have both _res.tif and _sunshad.tif
valid_image_numbers = {
    image_number for image_number, files in label_dict.items()
    if len(files) == 2 and any('_res.tif' in file
                                for file in files) and 
                                any('_sunshad.tif' in file for file in files)
}

# Filter original images and labels
filtered_original_images = [img for img in original_paths if
                             extract_image_number(img) in valid_image_numbers]
filtered_labels = [label for label in label_paths if
                    extract_image_number(label) in valid_image_numbers]

# List excluded images
excluded_images = [img for img in original_paths if 
                   extract_image_number(img) not in valid_image_numbers]

# # Print results
# print("Filtered Original Images:")
# print(filtered_original_images)
# print("\nFiltered Labels:")
# print(filtered_labels)
# print("\nExcluded Images:")
# print(excluded_images)
dataset = []
for path in filtered_original_images:
    print()
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image_number = os.path.basename(path).split('.')[0]
    label_paths = [path for path in filtered_labels
                    if path.endswith(f'{image_number}' + '_res.tif')
                    or path.endswith(f'{image_number}' + '_sunshad.tif')]
    res_label = cv2.imread(label_paths[0], cv2.IMREAD_UNCHANGED)
    res_label[res_label == 255] = 1
    sunshad_label = cv2.imread(label_paths[1], cv2.IMREAD_UNCHANGED)
    sunshad_label[sunshad_label == 255] = 1
    comb_label = 2 * res_label + sunshad_label
    features = get_features(bgr)

    res_label = res_label.ravel()
    sunshad_label = sunshad_label.ravel()
    comb_label = comb_label.ravel()

    dataset.append({"bgr": bgr, 
                    "features": features,
                    "res_label": res_label,
                    "sunshad_label": sunshad_label})


n_feat = features.shape[1]

feats_raw = []
comb_labels = []
for sample in dataset:
    feats_raw.append(sample["features"])
    comb_labels.append(sample["sunshad_label"])
del dataset

feats_raw = np.array(feats_raw).reshape((-1, n_feat)).astype(np.float32)
comb_labels = np.array(comb_labels).reshape((-1, 1)).astype(np.int32).ravel()
