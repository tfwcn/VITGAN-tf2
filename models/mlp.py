import tensorflow as tf
import sys

sys.path.append('')

from models.isn import ISN


class MLP(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        discriminator,
        dropout,
    ):
        super().__init__()
        self.discriminator = discriminator  # 是否鉴别器
        self.dn1 = tf.keras.layers.Dense(d_model*4, use_bias=False, activation='relu', kernel_regularizer='l2')
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dn2 = tf.keras.layers.Dense(d_model, use_bias=False, kernel_regularizer='l2')
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        if discriminator:
            self.dn1 = ISN(self.dn1)
            self.dn2 = ISN(self.dn2)

    def call(self, x, training):
        x = self.dn1(x)
        x = tf.nn.gelu(x)
        x = self.dropout1(x, training=training)
        x = self.dn2(x)
        x = self.dropout2(x, training=training)
        return x


if __name__ == "__main__":
    layer = MLP(
        d_model=768,
        discriminator=True,
        dropout=0.0,
    )
    x = tf.random.uniform([2, 5, 128], dtype=tf.float32)
    o1 = layer(x)
    tf.print('o1:', tf.shape(o1))
