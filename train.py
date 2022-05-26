import os
import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import sys

sys.path.append('')

from models.vitgan import ViTGAN
from datasets.dataset_creater import DatasetCreater
import  utils.tf_image_helper as tf_image_helper

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpu, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# 启动参数
parser = argparse.ArgumentParser()
parser.add_argument('--labels_dir', type=str,
    help='标签目录，只用放图片', default='./labels')
parser.add_argument('--models_dir', type=str,
    help='模型跟目录', default='./data/model/')
parser.add_argument('--batch_size', type=int,
    help='批次大小', default=2)
parser.add_argument('--image_size', type=int,
    help='图片大小', default=224)
parser.add_argument('--patch_size', type=int,
    help='块大小', default=16)
parser.add_argument('--overlapping', type=int,
    help='图像重叠部分', default=3)
argparse_args = parser.parse_args()
batch_size = argparse_args.batch_size
image_size = argparse_args.image_size
patch_size = argparse_args.patch_size
num_channels = 3
overlapping = argparse_args.overlapping # 图像重叠部分
d_model = patch_size * patch_size * num_channels
labels_dir = argparse_args.labels_dir
models_dir = argparse_args.models_dir

dataset_creater = DatasetCreater(
    labels_dir=labels_dir,
    batch_size=batch_size,
    image_size=image_size,
    patch_size=patch_size,
    num_channels=num_channels,
    d_model=d_model,
)
dataset = dataset_creater.get_dataset()


model = ViTGAN(
    image_size=image_size,
    patch_size=patch_size,
    num_channels=num_channels,
    overlapping=overlapping, # 图像重叠部分
    d_model=d_model,
    dropout=0.0,
    discriminator=True,
    k=1,
    k2=2,
)

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(loss=loss)
for x in dataset.take(1):
    tf.print('x:', tf.shape(x))
    noise = tf.random.normal([batch_size, 1, d_model], dtype=tf.float32)
    o = model(noise, training=True)
    tf.print('o:', tf.shape(o))
    o = model(noise, training=False)
    tf.print('o:', tf.shape(o))


checkpoint_dir = models_dir
if os.path.exists(checkpoint_dir):
    model.load_weights(checkpoint_dir)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    monitor='gen_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
    mode='min',
    save_freq='epoch',
)
# model_checkpoint_callback = tfa.callbacks.AverageModelCheckpoint(
#     update_weights=False,
#     filepath=checkpoint_dir,
#     monitor='gen_loss',
#     verbose=1,
#     save_best_only=False,
#     save_weights_only=False,
#     mode='min',
#     save_freq='epoch',
# )

class SaveImage(tf.keras.callbacks.Callback):
    def __init__(
        self,
        save_dir,
        batch_size,
        d_model,
        ):
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.d_model = d_model

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def on_epoch_end(self, epoch, logs=None):
        noise = tf.random.normal([self.batch_size, 1, self.d_model], dtype=tf.float32)
        imgs = self.model.generator(noise, training=False)
        for i in range(self.batch_size):
            img = imgs[i]
            tf_image_helper.save_image(img, os.path.join(self.save_dir,'%d_%d.jpg' % (epoch, i)))

save_dir = './data/output'
save_image_callback = SaveImage(
    save_dir=save_dir,
    batch_size=batch_size,
    d_model=d_model,
)

model.fit(
    dataset,
    epochs=10000,
    steps_per_epoch=1000,
    callbacks=[model_checkpoint_callback, save_image_callback]
)