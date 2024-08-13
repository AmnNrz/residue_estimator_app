import cv2
import os
import logging
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import tensorflow as tf

from image_data_generator import ImageDataGenerator
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from utils_segmentation import calculate_class_weights, confusion, extract_image_number, find_jpg_images, find_tif_labels, get_batch_features_and_labels, get_features

logging.basicConfig(level=logging.INFO)

##### Read an rgb image with its labels #####
path_to_data = ("C:/Users/brand/AgAID/residue_estimator/images/")

# Get the paths for all the original images and the tif labels
print('Finding original and label paths')
original_paths = find_jpg_images(path_to_data + "original/")
label_paths = find_tif_labels(path_to_data + "label/")

# Group labels by image number
label_dict = defaultdict(set)
for label in label_paths:
    image_number = extract_image_number(label)
    label_dict[image_number].add(label)

# Find valid image numbers that have both _res.tif and _sunshad.tif
valid_image_numbers = {
    image_number for image_number, files in label_dict.items()
    if len(files) == 2 and any('_res.tif' in file for file in files) and 
                                any('_sunshad.tif' in file for file in files)
}

# Create a dictionary of image numbers to paths, filtering by valid_image_numbers
img_dict = {
    image_number: path
    for path in original_paths
    if (image_number := extract_image_number(path)) in valid_image_numbers
}

# Remove invalid image numbers from label dict (that do not have _res.tif and _sunshad.tif)
invalid_image_numbers = [img_num for img_num in label_dict if img_num not in valid_image_numbers]
for img_num in invalid_image_numbers:
    label_dict.pop(img_num)

# Note that valid_image_numbers now map the valid numbers(ids) to the respective image paths and label paths

# Assuming `valid_image_numbers` is already filtered with valid images
valid_image_numbers = list(valid_image_numbers)

# Create lists of image paths and label paths
batch_size = 1  # Adjust as needed
n_features = 108  # Number of features, replace with the actual number

image_paths = [img_dict[img_num] for img_num in valid_image_numbers]
label_paths = [list(label_dict[img_num]) for img_num in valid_image_numbers]

# Initialize the data generator
train_data_generator = ImageDataGenerator(image_paths=image_paths, label_paths=label_paths, batch_size=batch_size, n_features=n_features, mode='train')
val_data_generator = ImageDataGenerator(image_paths=image_paths, label_paths=label_paths, batch_size=batch_size, n_features=n_features, mode='val')
test_data_generator = ImageDataGenerator(image_paths=image_paths, label_paths=label_paths, batch_size=batch_size, n_features=n_features, mode='test')

print('Calculating class weights')
class_weights = calculate_class_weights(label_paths)

print('Initializing the model')
# Define the neural network model using TensorFlow/Keras
model = Sequential()
model.add(Dense(128, input_dim=108, activation='relu')) # Input dimension = number of features, make sure to adjust accordingly
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model using the data generator
print('Training the model')
model.fit(
    train_data_generator,
    epochs=20,
    callbacks=[early_stopping],
    validation_data=val_data_generator,
    class_weight=class_weights
)

print('Testing the model on the test dataset')
predictions = model.predict(test_data_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Collect all the test labels for accuracy computation
all_test_labels = np.concatenate([test_data_generator[i][1] for i in range(len(test_data_generator))])

# Calculate the overall testing accuracy
accuracy = accuracy_score(all_test_labels, predicted_classes)
print(f'Overall Testing Accuracy: {accuracy}')

confusion(predicted_classes=predicted_classes, test_labels=all_test_labels)

print('Saving the model')
model.save('saved_models/model.keras')
# Save the model in HDF5 format
# model.save('saved_models/model1.h5')
# Export the model as saved model
model.export("saved_models/exported_model")
