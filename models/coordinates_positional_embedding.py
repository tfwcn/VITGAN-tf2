import tensorflow as tf


class CoordinatesPositionalEmbedding(tf.keras.layers.Layer):
    """
    博里叶位置编码
    """

    def __init__(
        self,
        patch_size,
        emb_dim,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.pos_emb = tf.keras.layers.Dense(emb_dim, use_bias=True)
        pos_input = tf.linspace(-1, 1, patch_size)
        pos_x, pos_y = tf.meshgrid(pos_input, pos_input)
        pos_input = tf.concat([pos_y[..., tf.newaxis],pos_x[..., tf.newaxis]], axis=-1)
        pos_input = tf.reshape(pos_input,[-1, 2])
        self.pos_input = pos_input

    def call(self, x):
        x_shape = tf.shape(x)
        batch_size = x_shape[0] * x_shape[1] # batch_size*num_patches
        x = self.pos_emb(self.pos_input) # (P*P, emb_dim)
        x = tf.math.sin(x)
        x = tf.repeat(x[tf.newaxis, ...], batch_size, axis=0) # (B*L, P*P, emb_dim)
        return x


if __name__ == "__main__":
    layer = CoordinatesPositionalEmbedding(
        patch_size=16,
        emb_dim=768 # 16*16*3
    )
    x = tf.random.uniform([2,196,768], dtype=tf.float32) # (batch_size*num_patches, patch_size*patch_size, emb_dim)
    o1 = layer(x)
    tf.print('o1:', tf.shape(o1))