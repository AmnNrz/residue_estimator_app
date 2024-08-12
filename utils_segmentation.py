import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2

import pandas as pd
import seaborn as sn
from skimage.feature import local_binary_pattern as LBP
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import tensorflow as tf

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

# Extract image numbers from file paths
def extract_image_number(file_path):
    base_name = os.path.basename(file_path)
    return base_name.split('_')[1].split('.')[0]

def localSD(mat, n):
    mat = np.float32(mat)
    mu = cv2.blur(mat, (n, n))
    mdiff = mu - mat
    mat2 = cv2.blur(np.float64(mdiff * mdiff), (n, n))
    sd = np.float32(cv2.sqrt(mat2))

    return sd

def calculate_combined_label(label_path_pair):
    # Find the path for _res.tif and _sunshad.tif labels
    res_label_path = next(path for path in label_path_pair if path.endswith(f'_res.tif'))
    sunshad_label_path = next(path for path in label_path_pair if path.endswith(f'_sunshad.tif'))

    res_label = cv2.imread(res_label_path, cv2.IMREAD_UNCHANGED)
    sunshad_label = cv2.imread(sunshad_label_path, cv2.IMREAD_UNCHANGED)

    # Convert to binary labels
    res_label[res_label == 255] = 1 # Nonresidue: 0, Residue: 1
    sunshad_label[sunshad_label == 255] = 1 # Shaded: 0 , Sunlit: 1
    return 2 * res_label + sunshad_label

def calculate_class_weights(label_paths):
    comb_labels = [] # 0-3, 4 classes total
    for label_path_pair in label_paths:

        comb_label = calculate_combined_label(label_path_pair=label_path_pair)
        comb_labels.append(comb_label)

    comb_labels = np.array(comb_labels).reshape((-1)).astype(np.int32)
    # Train = 0.64, Validation and Test = 0.36
    train_labels, _ = train_test_split(comb_labels, test_size=0.36, random_state=42)

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    print('Class weights:')
    print(class_weights)

    return class_weights

def get_features(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    l, a, bb = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(bgr)
    chan_funs = []
    img_size = b.shape

    ddepth = cv2.CV_16S

    for c in [b, g, r, h, s, v, l, a, bb]:
        chan_funs.append(c)
        chan_funs.append(cv2.GaussianBlur(c, (15, 15), cv2.BORDER_DEFAULT))
        chan_funs.append(cv2.GaussianBlur(c, (31, 31), cv2.BORDER_DEFAULT))
        chan_funs.append(localSD(c, 127))
        chan_funs.append(localSD(c, 63))
        chan_funs.append(localSD(c, 31))
        chan_funs.append(LBP(c, 32, 4, method="ror"))
        chan_funs.append(LBP(c, 24, 3, method="ror"))
        chan_funs.append(LBP(c, 16, 2, method="ror"))
        chan_funs.append(cv2.Laplacian(c, ddepth, ksize=3))
        chan_funs.append(cv2.Laplacian(c, ddepth, ksize=7))
        chan_funs.append(cv2.Laplacian(c, ddepth, ksize=15))
    ravels = []
    for cf in chan_funs:
        ravels.append(cf.ravel().T)
    feat = np.vstack(ravels).T
    return feat

# Return a list of raw features for each pixel, the combined label for each pixel, and the number of features
def get_batch_features_and_labels(filtered_original_images, filtered_labels):
    feats_raw, comb_labels = [], []
    for i, path in enumerate(filtered_original_images):
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        label_paths = filtered_labels[i]
        comb_label = calculate_combined_label(label_path_pair=label_paths)

        features = get_features(bgr)

        feats_raw.append(features)
        comb_labels.append(comb_label)
    
    return feats_raw, comb_label, features.shape[1]

def confusion(test_labels, predicted_classes):
    # Compute confusion matrix
    cm = tf.math.confusion_matrix(test_labels, predicted_classes).numpy()

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert to percentages
    cm_percent = cm_normalized * 100

    # Plot the normalized confusion matrix
    df_cm = pd.DataFrame(cm_percent, index=range(4), columns=range(4))
    sn.set_theme(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, fmt=".2f", annot_kws={"size": 16}, cmap='Blues')  # font size and color map
    plt.title('Normalized Confusion Matrix (Percentages)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()