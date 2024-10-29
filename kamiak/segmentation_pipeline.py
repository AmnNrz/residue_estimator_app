import cv2
import os
import logging
from collections import defaultdict
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import tensorflow as tf

from image_data_generator import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils_segmentation import calculate_class_weights, confusion, extract_image_number, find_jpg_images, find_tif_labels

logging.basicConfig(level=logging.INFO)



# Path to data
path_to_data = ("/home/a.norouzikandelati/Projects/"
                "residue_estimator_app/kamiak/data/images/")

# Find image and label paths
print('Finding original and label paths')
original_paths = find_jpg_images(path_to_data + "original/")
label_paths = find_tif_labels(path_to_data + "label/")

# Group labels by image number and find valid image numbers
label_dict = defaultdict(set)
for label in label_paths:
    image_number = extract_image_number(label)
    label_dict[image_number].add(label)

valid_image_numbers = {
    image_number for image_number, files in label_dict.items()
    if len(files) == 2 and any('_res.tif' in file for file in files) and 
                                any('_sunshad.tif' in file for file in files)
}

img_dict = {
    image_number: path
    for path in original_paths
    if (image_number := extract_image_number(path)) in valid_image_numbers
}

invalid_image_numbers = [img_num for img_num in label_dict if img_num not in valid_image_numbers]
for img_num in invalid_image_numbers:
    label_dict.pop(img_num)

valid_image_numbers = list(valid_image_numbers)

batch_size = 1  # Adjust as needed
n_features = 108  # Number of features, replace with the actual number

image_paths = [img_dict[img_num] for img_num in valid_image_numbers]
label_paths = [list(label_dict[img_num]) for img_num in valid_image_numbers]

# Optimizing Data Generator
# Initialize the training data generator and fit the scaler
train_data_generator = ImageDataGenerator(
    image_paths=image_paths,
    label_paths=label_paths,
    batch_size=batch_size,
    n_features=n_features,
    mode='train',
    shuffle=True
)

# Get the fitted scaler from the training generator
scaler = train_data_generator.scaler

# Initialize the validation and test data generators with the fitted scaler
val_data_generator = ImageDataGenerator(
    image_paths=image_paths,
    label_paths=label_paths,
    batch_size=batch_size,
    n_features=n_features,
    mode='val',
    scaler=scaler,
    shuffle=False
)

test_data_generator = ImageDataGenerator(
    image_paths=image_paths,
    label_paths=label_paths,
    batch_size=batch_size,
    n_features=n_features,
    mode='test',
    scaler=scaler,
    shuffle=False
)

print('Calculating class weights')
class_weights = calculate_class_weights(label_paths)

# Setup strategy for multi-worker training with MirroredStrategy
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("CUDA version", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN version", tf.sysconfig.get_build_info()['cudnn_version'])

from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers import Adam

# Enable mixed precision
set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    print('Initializing the model')
    # Define the model within the strategy scope
    model = Sequential()
    model.add(Dense(128, input_dim=n_features, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax', dtype='float32'))

    # Compile the model
    model.compile(
        optimizer='adam',
          loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
        )


# Time 
start_time = time.time()

# Train the model using the data generator
print('Training the model')
model.fit(
    train_data_generator,
    epochs=20,
    callbacks=[early_stopping],
    validation_data=val_data_generator,
    class_weight=class_weights
)

end_time = time.time()

training_time = end_time - start_time

# Convert the time into appropriate units (seconds, minutes, or hours)
if training_time < 60:
    print(f'Training completed in {training_time:.2f} seconds.')
elif training_time < 3600:
    minutes, seconds = divmod(training_time, 60)
    print(f'Training completed in {int(minutes)} minutes and {int(seconds)} seconds.')
else:
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'Training completed in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.')

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
