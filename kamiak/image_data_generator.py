import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from utils_segmentation import get_batch_features_and_labels
import tensorflow as tf

class ImageDataGenerator(Sequence):
    def __init__(self, image_paths, label_paths,
                  batch_size, n_features, mode,
                    scaler=None, shuffle=True,
                      num_parallel_calls=tf.data.AUTOTUNE, **kwargs):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.n_features = n_features
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.mode = mode
        self.shuffle = shuffle
        self.num_parallel_calls = num_parallel_calls
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_label_paths = self.label_paths[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        feats, labels = self.__data_generation(batch_image_paths, batch_label_paths)
        return feats, labels

    def on_epoch_end(self):
        # Shuffle data at the end of each epoch if required
        if self.shuffle:
            combined = list(zip(self.image_paths, self.label_paths))
            np.random.shuffle(combined)
            self.image_paths, self.label_paths = zip(*combined)

    def __data_generation(self, batch_image_paths, batch_label_paths):
        # Generate data for the batch
        feats_raw, comb_labels, _ = get_batch_features_and_labels(
            filtered_original_images=batch_image_paths,
            filtered_labels=batch_label_paths
        )
        
        # Reshape and standardize features
        feats_raw = np.array(feats_raw).reshape((-1, self.n_features)).astype(np.float32)
        comb_labels = np.array(comb_labels).reshape((-1)).astype(np.int32)

        if self.mode == 'train':
            # Fit and transform the training features
            train_feats_scaled = self.scaler.fit_transform(feats_raw)
            return train_feats_scaled.astype(np.float16), comb_labels
        
        else:
            # For validation and test, just transform using the fitted scaler
            feats_scaled = self.scaler.transform(feats_raw)
            return feats_scaled.astype(np.float16), comb_labels
