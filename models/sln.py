import tensorflow as tf


class SLN(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = tf.keras.layers.Dense(d_model, use_bias=False)
        self.beta = tf.keras.layers.Dense(d_model, use_bias=False)
        self.ln = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-6,
            center=False,
            scale=False,
        )

    def call(self, h, w, training):
        x = self.gamma(w) * self.ln(h, training=training) * self.beta(w)
        return x

if __name__ == "__main__":
    layer = SLN(128)
    h = tf.random.uniform([2,5,128], dtype=tf.float32)
    w = tf.random.uniform([2,5,128], dtype=tf.float32)
    o1 = layer(h, w, training=True)
    tf.print('o1:', o1)