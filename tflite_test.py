

# Load the TFLite model
from collections import defaultdict
import glob
import os
import sys

import cv2
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf

from utils_segmentation import get_features


tflite_model_file = './saved_models/exported_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)

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

print('Generating raw features and labels')
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

# Reshape and type conversion
feats_raw = np.array(feats_raw).reshape((-1, n_feat)).astype(np.float32)

# Reshape comb_labels to a compatible shape
comb_labels = np.array(comb_labels).reshape((-1)).astype(np.int32)

print('Standardizing features')
# Standardize the data
scaler = StandardScaler()
test_feats_scaled = scaler.fit_transform(feats_raw)

# Load the TFLite model
tflite_model_file = './saved_models/exported_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)

interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# Test the TFLite model on the test data
input_shape = input_details[0]['shape']
input_data = np.array(test_feats_scaled, dtype=np.float32)
if input_data.shape[1] == input_shape[1]:  # Assuming input_data has shape (num_samples, num_features)
    input_data = input_data.reshape((-1, input_shape[1]))
else:
    raise ValueError(f"Unexpected input shape: {input_data.shape}. Expected shape: {input_shape}")

predictions = []
print("Running inference")
print(input_data.shape)
for i in range(input_data.shape[0]):
    interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1])  # Use i:i+1 to preserve batch dimension
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions.append(output_data)

print('Calculating results')
np.set_printoptions(threshold=sys.maxsize)
print(len(predictions))
print(comb_labels)