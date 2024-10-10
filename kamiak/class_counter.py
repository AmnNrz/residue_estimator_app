import cv2
import os
import logging
import glob
from collections import defaultdict
import numpy as np

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

feats_raw, comb_labels = [], []
comb_label_count = defaultdict(int)
total_pixels = 0
for path in filtered_original_images:
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

    comb_labels.append(comb_label)

    # Count occurrences of each class
    unique, counts = np.unique(comb_label, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    for k, v in class_counts.items():
        comb_label_count[k] += v
    total_pixels += comb_label.size

# Calculate percentages
comb_label_percentage = {k: (v / total_pixels) * 100 for k, v in comb_label_count.items()}

# Print the combined label counts
print("Combined Label Counts:", comb_label_count)
print("Combined Label Percentages:", comb_label_percentage)