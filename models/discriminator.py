import tensorflow as tf
import sys


sys.path.append('')

from models.patch_embedding import PatchEmbedding
from models.discriminator_transformer_encoder import DiscriminatorEncoder
from models.mlp import MLP
from models.positional_embedding import PositionalEmbedding


class Discriminator(tf.keras.layers.Layer):
    """
    鉴别器
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        overlapping=3,
        d_model=768,
        out_dim=1,
        dropout=0.0,
        discriminator=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.overlapping = overlapping
        self.d_model = d_model
        self.out_dim = out_dim
        self.dropout = dropout
        self.discriminator = discriminator
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            overlapping=overlapping,
            emb_dim=d_model,
            discriminator=discriminator,
        )
        # 输入位置编码
        self.patch_positional_embedding = PositionalEmbedding(
            sequence_length=self.num_patches+1,
            emb_dim=self.d_model,
        )
        self.discriminator_transformer_encoder = DiscriminatorEncoder(
            self.d_model,
            num_heads=8,
            num_layers=4,
            dropout=dropout,
            discriminator=discriminator,
        )
        self.mlp = MLP(out_dim, discriminator=discriminator, dropout=0.0)
        self.cls_token = tf.Variable(tf.random.uniform([1, 1, self.d_model], dtype=tf.float32), dtype=tf.float32)

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.patch_embedding(x)
        cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        x = tf.concat([cls_token, x], axis=1)
        x_pos = self.patch_positional_embedding()
        x += x_pos
        x = self.discriminator_transformer_encoder(x, training=training)
        x = self.mlp(x)
        x = x[:,0,:]
        # x = tf.math.sigmoid(x)
        return x


if __name__ == "__main__":
    layer = Discriminator(
        image_size=224,
        patch_size=16,
        num_channels=3,
        d_model=768
    )
    x = tf.random.uniform([2,224,224,3], dtype=tf.float32)
    o1 = layer(x, training=True)
    tf.print('o1:', tf.shape(o1))
    o1 = layer(x, training=False)
    tf.print('o1:', tf.shape(o1))