import cv2
import os
import logging
import glob
import seaborn as sn
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import tensorflow as tf; tf.keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from utils_segmentation import confusion, get_features


logging.basicConfig(level=logging.INFO)

##### Read an rgb image with its labels #####
path_to_data = ("C:/Users/brand/AgAID/residue_estimator/images/")

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

print('Generating features and getting labels')
feats_raw, comb_labels = [], []
for path in filtered_original_images:
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image_number = os.path.basename(path).split('.')[0]
    label_paths = [path for path in filtered_labels
                    if path.endswith(f'{image_number}' + '_res.tif')
                    or path.endswith(f'{image_number}' + '_sunshad.tif')]
    res_label = cv2.imread(label_paths[0], cv2.IMREAD_UNCHANGED)
    sunshad_label = cv2.imread(label_paths[1], cv2.IMREAD_UNCHANGED)
    # Convert to binary labels
    res_label[res_label == 255] = 1 # Nonresidue: 0, Residue: 1
    sunshad_label[sunshad_label == 255] = 1 # Shaded: 0 , Sunlit: 1
    comb_label = 2 * res_label + sunshad_label
    features = get_features(bgr)

    feats_raw.append(features)
    comb_labels.append(comb_label)

    
n_feat = features.shape[1]

print('Reshaping features and lable arrays')
# Reshape and type conversion
feats_raw = np.array(feats_raw).reshape((-1, n_feat)).astype(np.float32)

# Reshape comb_labels to a compatible shape
comb_labels = np.array(comb_labels).reshape((-1)).astype(np.int32)

print('Creating data splits')
# Split the data
train_feats, test_feats, train_labels, test_labels = train_test_split(
    feats_raw, comb_labels, test_size=0.2, random_state=42
)

print('Standardizing data')
# Standardize the data
scaler = StandardScaler()
train_feats_scaled = scaler.fit_transform(train_feats)
test_feats_scaled = scaler.transform(test_feats)

print('Calculating class weights')
# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
print('Class weights:')
print(class_weights)

print('Initializing the model')
# Define the neural network model using TensorFlow/Keras
model = Sequential()
model.add(Dense(128, input_dim=train_feats_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print('Training the model')
# Train the model
history = model.fit(train_feats_scaled, train_labels, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)

print('Saving the model')
# Save the model in the TensorFlow keras format
model.save('saved_models/model.keras')
# Save the model in HDF5 format
# model.save('saved_models/model1.h5')
# Export the model as saved model
model.export("saved_models/exported_model")

print('Testing the model')
# Predict on the test data
predictions = model.predict(test_feats_scaled)
predicted_classes = np.argmax(predictions, axis=1)

confusion(predicted_classes=predicted_classes, test_labels=test_labels)




