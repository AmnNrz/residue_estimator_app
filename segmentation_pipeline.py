# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
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
##### Read an rgb image with its labels
path_to_data = ("/mnt/C250892050891BF3/Projects/residue_estimator/images/")

### Function to find images
def find_jpg_images(root_dir):
    jpg_image = []
    for dir_name,a,b in os.walk(root_dir):
        search_pattern = os.path.join(dir_name, '*.jpg')
        for filename in glob.glob(search_pattern):
            print(filename)
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

### Read original image
path_to_original = path_to_data + "original/"
original_files = os.listdir(path_to_original)

### Read labels
path_to_labels = path_to_data + "label/"
find_tif_labels(path_to_labels)







