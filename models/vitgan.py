import tensorflow as tf
import tensorflow_addons as tfa
import sys

sys.path.append('')

from models.discriminator import Discriminator
from models.generator import Generator


class ViTGAN(tf.keras.Model):
    """
    ViTGAN
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        overlapping=3, # 图像重叠部分
        d_model=768,
        dropout=0.0,
        discriminator=True,
        k=3,
        k2=1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.overlapping = overlapping
        self.d_model = d_model
        self.dropout = dropout
        self.k = k
        self.k2 = k2
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        # 生成器
        self.generator = Generator(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            d_model=d_model,
            dropout=dropout,
        )
        # 鉴别器
        self.discriminator = Discriminator(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            overlapping=overlapping,
            d_model=d_model,
            out_dim=128,
            dropout=dropout,
            discriminator=discriminator,
        )
        # self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # self.generator_optimizer = tfa.optimizers.MovingAverage(self.generator_optimizer, average_decay = 0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # self.discriminator_optimizer = tfa.optimizers.MovingAverage(self.discriminator_optimizer, average_decay = 0.999)

    def call(self, x, training):
        # batch_size = tf.shape(x)[0]
        x = self.generator(x, training=training)
        x = self.discriminator(x, training=training)
        return x

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, data):
        images = data
        batch_size= tf.shape(images)[0]
        # 先训练k步鉴别器
        for _ in range(self.k):
            noise = tf.random.normal([batch_size, 1, self.d_model], dtype=tf.float32)

            # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                # gen_loss = self.generator_loss(fake_output)
                disc_loss = self.discriminator_loss(real_output, fake_output)

            # gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        for _ in range(self.k2):
            # 再训练生成器
            noise = tf.random.normal([batch_size, 1, self.d_model], dtype=tf.float32)

            with tf.GradientTape() as gen_tape:
                generated_images = self.generator(noise, training=True)

                fake_output = self.discriminator(generated_images, training=True)

                gen_loss = self.generator_loss(fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return {'gen_loss': gen_loss, 'disc_loss:': disc_loss}

    @tf.function
    def test_step(self, data):
        images = data
        batch_size= tf.shape(images)[0]
        noise = tf.random.normal([batch_size, 1, self.d_model], dtype=tf.float32)
        generated_images = self.generator(noise, training=True)

        real_output = self.discriminator(images, training=True)
        fake_output = self.discriminator(generated_images, training=True)

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)
        return {'gen_loss': gen_loss, 'disc_loss:': disc_loss}

    @tf.function
    def predict_step(self, data):
        images = data
        batch_size= tf.shape(images)[0]
        noise = tf.random.normal([batch_size, 1, self.d_model], dtype=tf.float32)
        generated_images = self.generator(noise, training=True)
        return generated_images


if __name__ == "__main__":
    layer = ViTGAN(
        image_size=224,
        patch_size=16,
        num_channels=3,
        overlapping=3, # 图像重叠部分
        d_model=768,
        dropout=0.0,
    )
    x = tf.random.uniform([2,1,768], dtype=tf.float32)
    y = tf.random.uniform([2,224,224,3], dtype=tf.float32)
    o1 = layer(x, training=True)
    tf.print('o1:', tf.shape(o1))
    o1 = layer(x, training=False)
    tf.print('o1:', tf.shape(o1))