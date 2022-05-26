import tensorflow as tf

import sys

sys.path.append('')

from models.msa import MSA
from models.mlp import MLP
from models.sln import SLN


class GeneratorEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.sln1 = SLN(d_model)
        self.msa1 = MSA(d_model, num_heads, discriminator=False)
        self.sln2 = SLN(d_model)
        self.mlp1 = MLP(d_model, discriminator=False, dropout=dropout)

    def call(self, x, w, training):
        h = x
        x = self.sln1(h=x, w=w, training=training)
        x = self.msa1(v=x, k=x, q=x, mask=None)
        x = x + h
        h = x
        x = self.sln2(h=x, w=w, training=training)
        x = self.mlp1(x)
        x = x + h
        return x

class GeneratorEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.encoder_layers = [GeneratorEncoderLayer(d_model, num_heads, dropout=dropout) for i in range(num_layers)]

    def call(self, x, w, training):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x=x, w=w, training=training)
        return x

if __name__ == "__main__":
    # layer = EncoderLayer(256, 8)
    layer = GeneratorEncoder(256, 8, 4)
    x = tf.random.uniform([2,5,256], dtype=tf.float32)
    w = tf.random.uniform([2,5,256], dtype=tf.float32)
    o1 = layer(x, w, training=True)
    tf.print('o1:', tf.shape(o1))