import numpy as np
import tensorflow as tf


class MappingNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, lrmul=0.01):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.lrmul = lrmul
        self.ln = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-6,
            center=False,
            scale=False,
        )
        self.dn_layers = []
        for _ in range(num_layers):
            self.dn_layers.append(tf.keras.layers.Dense(d_model, use_bias=True, kernel_initializer='he_uniform', kernel_regularizer='l2'))
            self.dn_layers.append(tf.keras.layers.LeakyReLU(lrmul))
        self.net = tf.keras.Sequential(self.dn_layers)

    def call(self, x, training):
        x = self.ln(x, training=training)
        x = self.net(x)
        return x

    def get_config(self):
        config = super(MappingNetwork, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'lrmul': self.lrmul,
        })
        return config

if __name__ == "__main__":
    layer = MappingNetwork(256, 8)
    x = tf.random.uniform([2,1,256], dtype=tf.float32)
    o1 = layer(x)
    tf.print('o1:', tf.shape(o1))