import tensorflow as tf

import sys

sys.path.append('')

from models.msa import MSA
from models.mlp import MLP


class DiscriminatorEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.0, discriminator=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.discriminator = discriminator
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.msa1 = MSA(d_model, num_heads, discriminator=discriminator)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp1 = MLP(d_model, discriminator=discriminator, dropout=dropout)

    def call(self, x, training):
        h = x
        x = self.ln1(x, training=training)
        x = self.msa1(v=x, k=x, q=x, mask=None)
        x = x + h
        h = x
        x = self.ln2(x, training=training)
        x = self.mlp1(x)
        x = x + h
        return x

class DiscriminatorEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.0, discriminator=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.discriminator = discriminator
        self.encoder_layers = [DiscriminatorEncoderLayer(d_model, num_heads, dropout=dropout, discriminator=discriminator) for i in range(num_layers)]

    def call(self, x, training):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x=x, training=training)
        return x

if __name__ == "__main__":
    # layer = DiscriminatorEncoderLayer(256, 8)
    layer = DiscriminatorEncoder(256, 8, 4)
    x = tf.random.uniform([2,5,256], dtype=tf.float32)
    o1 = layer(x, training=True)
    tf.print('o1:', tf.shape(o1))