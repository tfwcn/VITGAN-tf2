import tensorflow as tf


class PositionalEmbedding(tf.Module):
    """
    输入位置编码
    """

    def __init__(
        self,
        sequence_length,
        emb_dim,
        name=None,
    ):
        super().__init__(name=name)
        self.emb_dim = emb_dim
        self.sequence_length = sequence_length
        self.pos_emb = tf.keras.layers.Dense(emb_dim, use_bias=False)
        self.pos_input = tf.linspace(-1, 1, sequence_length)[tf.newaxis, :, tf.newaxis]

    def __call__(self):
        x = self.pos_emb(self.pos_input)
        x = tf.math.sin(x)
        return x


if __name__ == "__main__":
    layer = PositionalEmbedding(
        sequence_length=196,
        emb_dim=768
    )
    o1 = layer()
    tf.print('o1:', tf.shape(o1))