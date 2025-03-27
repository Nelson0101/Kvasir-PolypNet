import os
import cv2
from data.PolypDataset import  PolypDataset

class DataPreprocessor:
    def __init__(self, main_path):
        self.main_path = main_path
        self.train_images, self.train_labels, self.val_images, self. val_labels = self._load_data()

    def create_dataset(self, type, transfrom=None)->PolypDataset:

        if type =="train":
            return PolypDataset(self.train_images, self.train_labels, transfrom)
        elif type == "val":
            return PolypDataset(self.val_images, self.val_labels, transfrom)
        else:
            raise ValueError("type must be 'train' or 'val'")


    def _load_data(self):
        train_images_polyp = []
        train_images_normal = []
        val_images_normal = []
        val_images_polyp = []

        print(f"Loading data from: {self.main_path}")
        for subdirectory in os.listdir(self.main_path):
            if subdirectory == ".DS_Store":
                continue
            is_val = subdirectory == "val"
            subdirectory_path = os.path.join(self.main_path, subdirectory)

            print(f"Scanning subdirectory: {subdirectory_path}")
            for sub_subdirectory in os.listdir(subdirectory_path):
                if sub_subdirectory == ".DS_Store":
                    continue
                sub_subdirectory_path = os.path.join(subdirectory_path, sub_subdirectory)

                print(f"Processing: {sub_subdirectory_path}")
                for filename in os.listdir(sub_subdirectory_path):
                    img_path = os.path.join(sub_subdirectory_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load: {img_path}")
                        continue

                    if sub_subdirectory == "polyps":
                        (val_images_polyp if is_val else train_images_polyp).append(img)
                    elif sub_subdirectory == "normal-cecum":
                        (val_images_normal if is_val else train_images_normal).append(img)
                    else:
                        print(f"Unexpected subdirectory: {sub_subdirectory}")

        train_images = train_images_polyp + train_images_normal
        val_images = val_images_polyp + val_images_normal
        train_labels = [1] * len(train_images_polyp) + [0] * len(train_images_normal)
        val_labels = [1] * len(val_images_polyp) + [0] * len(val_images_normal)

        print(f"Loaded - Train: {len(train_images)} images, Val: {len(val_images)} images")
        return train_images, train_labels, val_images, val_labels