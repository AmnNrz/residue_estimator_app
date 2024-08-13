import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import PyDataset # type:ignore
from sklearn.preprocessing import StandardScaler
from utils_segmentation import get_batch_features_and_labels

class ImageDataGenerator(PyDataset):
    def __init__(self, image_paths, label_paths, batch_size, n_features, mode, **kwargs):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.on_epoch_end()
        self.mode = mode

        # Call the parent constructor
        super().__init__(**kwargs)

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_label_paths = self.label_paths[index * self.batch_size:(index + 1) * self.batch_size]

        print(batch_image_paths)
        print(batch_label_paths)

        # Generate data
        feats, labels = self.__data_generation(batch_image_paths, batch_label_paths)
        return feats, labels

    def on_epoch_end(self):
        # Optional: shuffle data at the end of each epoch
        # combined = list(zip(self.image_paths, self.label_paths))
        # np.random.shuffle(combined)
        # self.image_paths, self.label_paths = zip(*combined)
        pass

    def __data_generation(self, batch_image_paths, batch_label_paths):
        # Generate data for the batch
        feats_raw, comb_labels, _ = get_batch_features_and_labels(
            filtered_original_images=batch_image_paths,
            filtered_labels=batch_label_paths
        )
        # Reshape and standardize features
        feats_raw = np.array(feats_raw).reshape((-1, self.n_features)).astype(np.float32)
        comb_labels = np.array(comb_labels).reshape((-1)).astype(np.int32)

        # Split the data
        train_feats, test_feats, train_labels, test_labels = train_test_split(
            feats_raw, comb_labels, test_size=0.2, random_state=42
        )
        train_feats, val_feats, train_labels, val_labels = train_test_split(
           train_feats, train_labels, test_size=0.2, random_state=42
        )

        if self.mode == 'train':
            train_feats_scaled = self.scaler.fit_transform(train_feats)
            return train_feats_scaled, train_labels
    
        elif self.mode == 'val':
            val_feats_scaled = self.scaler.fit_transform(val_feats)
            return val_feats_scaled, val_labels
        
        else:
            test_feats_scaled = self.scaler.fit_transform(test_feats)
            return test_feats_scaled, test_labels
