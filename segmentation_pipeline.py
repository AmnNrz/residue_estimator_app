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

# Reshape and type conversion
feats_raw = np.array(feats_raw).reshape((-1, n_feat)).astype(np.float32)

# Reshape comb_labels to a compatible shape
comb_labels = np.array(comb_labels).reshape((-1)).astype(np.int32)
# -

filtered_original_images

# +
from sklearn.decomposition import IncrementalPCA

# Split the data
train_feats, test_feats, train_labels, test_labels = train_test_split(
    feats_raw, comb_labels, test_size=0.2, random_state=42
)

# Standardize the data
scaler = StandardScaler()
train_feats_scaled = scaler.fit_transform(train_feats)

# Apply Incremental PCA
n_components = 2  # Or any other suitable value
batch_size = 1000  # Adjust the batch size according to your memory capacity
incremental_pca = IncrementalPCA(n_components=n_components)

for i in range(0, train_feats_scaled.shape[0], batch_size):
    incremental_pca.partial_fit(train_feats_scaled[i:i+batch_size])

train_feats_scaled_pca = np.zeros((train_feats_scaled.shape[0], incremental_pca.n_components_))

for i in range(0, train_feats_scaled.shape[0], batch_size):
    train_feats_scaled_pca[i:i+batch_size] = incremental_pca.transform(train_feats_scaled[i:i+batch_size])

# +
clf = MLPClassifier(max_iter=1000)
n_feat = train_feats_scaled_pca.shape[1]
# tune hyperparameters
print(n_feat)

layers = []
for layer1 in [2, 4]:
    for layer2 in [4, 8]:
        layer = (
            int(layer1 * n_feat),
            int(np.sqrt(layer1 * n_feat * layer2 * n_components)),
            layer2 * n_components,
        )
        layers.append(layer)
print(layers)

parameters = {"hidden_layer_sizes": layers, "activation": ("relu", "logistic")}

# +
# search = HalvingGridSearchCV(
#     clf, parameters, n_jobs=-1, cv=5, verbose=3, aggressive_elimination=True
# )

# +
# search.fit(train_feats_scaled_pca, train_labels)

# +
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)
# pipeline = Pipeline(
#     steps=[("scaler", scaler), ("pca", incremental_pca), ("clf", search.best_estimator_)]
# )

# # filename = os.path.join(
# #     p3, "model_pipeline_" + version + "_" + model + "_" + hy + "_final.pk.sav"
# # )
# # with open(filename, "wb") as f:  # Python 3: open(..., 'wb'
# #     pickle.dump(pipeline, f)

# pred = pipeline.predict(test_feats)

# M, f, a = cornfusion(test_labels, pred, n_components)

# plt.matshow(M) 
# plt.ylabel("Predicted")
# plt.xlabel("Observed")
# plt.title(" Confusion Matrix")
# -

# # Partial fit 

# +
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
# Create a wrapper for MLPClassifier to handle 

class IncrementalMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                 max_iter=200, shuffle=True, random_state=None, tol=1e-4, verbose=False,
                 warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10,
                 max_fun=15000):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.classes_ = None
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver,
                                   alpha=self.alpha, batch_size=self.batch_size, learning_rate=self.learning_rate,
                                   learning_rate_init=self.learning_rate_init, power_t=self.power_t, max_iter=self.max_iter,
                                   shuffle=self.shuffle, random_state=self.random_state, tol=self.tol, verbose=self.verbose,
                                   warm_start=self.warm_start, momentum=self.momentum, nesterovs_momentum=self.nesterovs_momentum,
                                   early_stopping=self.early_stopping, validation_fraction=self.validation_fraction,
                                   beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, n_iter_no_change=self.n_iter_no_change,
                                   max_fun=self.max_fun)
    
    def partial_fit(self, X, y, classes=None):
        if self.classes_ is None:
            self.classes_ = np.unique(y) if classes is None else classes
        self.model.partial_fit(X, y, classes=self.classes_)
        return self
    
    def fit(self, X, y):
        for _ in range(10):  # Example: 10 epochs
            self.partial_fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)





# Use HalvingGridSearchCV
search = HalvingGridSearchCV(
    IncrementalMLPClassifier(),
    parameters,
    cv=5,  # 3-fold cross-validation
    verbose=3,
    n_jobs=-1,
    aggressive_elimination=True
)


# Fit the model using HalvingGridSearchCV
search.fit(train_feats_scaled_pca, train_labels)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
pipeline = Pipeline(
    steps=[("scaler", scaler), ("pca", incremental_pca), ("clf", search.best_estimator_)]
)

# filename = os.path.join(
#     p3, "model_pipeline_" + version + "_" + model + "_" + hy + "_final.pk.sav"
# )
# with open(filename, "wb") as f:  # Python 3: open(..., 'wb'
#     pickle.dump(pipeline, f)

pred = pipeline.predict(test_feats)

M, f, a = cornfusion(test_labels, pred, n_components)

plt.matshow(M) 
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.title(" Confusion Matrix")
