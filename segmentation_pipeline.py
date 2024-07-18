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
import tensorflow as tf; tf.keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from utils_segmentation import get_features


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

dataset = []
for path in filtered_original_images:
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image_number = os.path.basename(path).split('.')[0]
    label_paths = [path for path in filtered_labels
                    if path.endswith(f'{image_number}' + '_res.tif')
                    or path.endswith(f'{image_number}' + '_sunshad.tif')]
    res_label = cv2.imread(label_paths[0], cv2.IMREAD_UNCHANGED)
    sunshad_label = cv2.imread(label_paths[1], cv2.IMREAD_UNCHANGED)
    # Convert to binary labels
    res_label[res_label == 255] = 1
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

# Split the data
train_feats, test_feats, train_labels, test_labels = train_test_split(
    feats_raw, comb_labels, test_size=0.2, random_state=42
)

# Standardize the data
scaler = StandardScaler()
train_feats_scaled = scaler.fit_transform(train_feats)
test_feats_scaled = scaler.transform(test_feats)

# Define the neural network model using TensorFlow/Keras
model = Sequential()
model.add(Dense(128, input_dim=train_feats_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_feats_scaled, train_labels, epochs=1, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping])

# Save the model in the TensorFlow keras format
model.save('saved_models/model.keras')
# Save the model in HDF5 format
# model.save('saved_models/model1.h5')
# Export the model as saved model
model.export("saved_models/exported_model")

# Predict on the test data
predictions = model.predict(test_feats_scaled)
predictions = (predictions > 0.5).astype(int)

# Compute confusion matrix
cm = tf.math.confusion_matrix(test_labels, predictions).numpy()

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Convert to percentages
cm_percent = cm_normalized * 100

# Plot the normalized confusion matrix
df_cm = pd.DataFrame(cm_percent, index=range(2), columns=range(2))
sn.set_theme(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, fmt=".2f", annot_kws={"size": 16}, cmap='Blues')  # font size and color map
plt.title('Normalized Confusion Matrix (Percentages)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


