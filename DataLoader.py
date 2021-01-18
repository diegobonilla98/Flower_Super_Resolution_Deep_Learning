import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np


class DataLoader:
    def __init__(self, data_path, target_size, size_diff_factor: int):
        self.size_diff_factor = size_diff_factor
        self.data_path = data_path
        self.target_size = target_size
        self.all_images = np.array(glob.glob(os.path.join(data_path, '*')))
        self.test_images = self.all_images[-3:]
        self.all_images = self.all_images[:-3]
        self.num_images = len(self.all_images)
        print(f'Found dataset with {self.num_images} images.')

    def load_image(self, path, loading: str, is_rgb=False):
        image = cv2.imread(path)
        if is_rgb:
            image = image[:, :, ::-1]
        if loading.lower() == 'y':
            image = cv2.resize(image, self.target_size)
        elif loading.lower() == 'x':
            image = cv2.resize(image, (self.target_size[0] // self.size_diff_factor,
                                       self.target_size[1] // self.size_diff_factor))
            image = cv2.resize(image, self.target_size)
        return (image.astype('float32') - 127.5) / 127.5

    def load_batch(self, batch_size):
        indexes = np.random.choice(self.num_images, batch_size, replace=False)
        # indexes = np.random.randint(low=0, high=self.num_images - 1, size=(batch_size,)).astype(int)
        batch_y = np.array([self.load_image(p, loading='y') for p in self.all_images[indexes]], 'float32')
        batch_x = np.array([self.load_image(p, loading='x') for p in self.all_images[indexes]], 'float32')
        return batch_x, batch_y
