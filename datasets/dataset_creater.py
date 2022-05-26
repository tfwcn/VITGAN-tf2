import tensorflow as tf
import numpy as np
import random
import sys

sys.path.append('')

import utils.file_helper as file_helper
import  utils.tf_image_helper as tf_image_helper

class DatasetCreater():
    def __init__(
        self,
        labels_dir,
        batch_size,
        image_size=224,
        patch_size=16,
        num_channels=3,
        d_model=768,
        ):
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.d_model = d_model
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.get_images_path()

    def get_images_path(self):
        self.labels = file_helper.read_file_list(self.labels_dir, pattern = r'[\.png|\.jpg]')
        print('图片数:', len(self.labels))
    
    def generator(self):
        labels_clone = self.labels.copy()
        while True:
            random.shuffle(labels_clone)
            for label in labels_clone:
                yield label
    
    @tf.function
    def convert_image(self, path):
        img = tf_image_helper.read_image(path)
        img = tf_image_helper.resize_image(img, self.image_size, self.image_size)
        return img

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string)
            ) # 最后一个后面不能有逗号
        )
        dataset = dataset.map(self.convert_image)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    labels_dir = 'Z:\\Labels\\lfw\\lfw_mtcnnpy_182'
    dataset_creater = DatasetCreater(
        labels_dir=labels_dir,
        batch_size=8,
        image_size=224,
        patch_size=16,
        num_channels=3,
        d_model=768,
    )
    dataset = dataset_creater.get_dataset()
    for x in dataset.take(2):
        tf.print('x:', tf.shape(x))