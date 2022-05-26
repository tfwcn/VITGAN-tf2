import tensorflow as tf
import numpy as np
import math


class ModulatedLinear(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        demodulation=True,
        use_bias=False,
        kernel_initializer=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.demodulation = demodulation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

    def build(self, inputs_shape):
        e_fou_shape, style_shape = inputs_shape
        # print('e_fou_shape:', e_fou_shape)
        # print('style_shape:', style_shape)
        self.scale = 1 / math.sqrt(self.hidden_dim)
        self.weight = self.add_weight(
            name='w',
            shape=[1,self.hidden_dim,self.output_dim],
            dtype=tf.float32,
            # initializer=tf.initializers.GlorotNormal(),
            initializer=self.kernel_initializer,
        )
        self.style_to_mod = tf.keras.layers.Dense(
            self.hidden_dim, 
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )

    def call(self, inputs):
        '''
        input: (B*L,P*P,E)
        style: (B, L, E=P*P*C)
        '''
        assert isinstance(inputs, tuple) and len(inputs) == 2
        x, style = inputs
        batch_size = tf.shape(x)[0] # B*L
        # Computing the weight
        style = self.style_to_mod(style)
        style = tf.reshape(style, [batch_size, self.hidden_dim, 1]) # (B*L, hidden_dim, 1)
        weight = self.scale * self.weight * style # (B*L, hidden_dim, output_dim)
        weight = weight / (tf.math.reduce_euclidean_norm(weight, axis=1, keepdims=True) + 1e-8) # [B*L, hidden_dim, output_dim]

        # Computing the out
        x = tf.matmul(x, weight) # (B*L, P*P, output_dim)
        return x


if __name__ == "__main__":
    output_dim = 3 # 16*16*3
    hidden_dim = 768 # 16*16*3
    kernel_initializer = tf.random_uniform_initializer(
        -1/output_dim,
        1/output_dim
    )
    layer = ModulatedLinear(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    e_fou = tf.random.uniform([2*196,256,768], dtype=tf.float32)
    y = tf.random.uniform([2,196,768], dtype=tf.float32)
    o1 = layer((e_fou, y))
    tf.print('o1:', tf.shape(o1))