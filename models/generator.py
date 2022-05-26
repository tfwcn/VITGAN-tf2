import tensorflow as tf
import sys

sys.path.append('')

from models.mapping_network import MappingNetwork
from models.generator_transformer_encoder import GeneratorEncoder
from models.coordinates_positional_embedding import CoordinatesPositionalEmbedding
from models.siren import Siren
from models.sln import SLN
from models.positional_embedding import PositionalEmbedding
from models.modulated_linear import ModulatedLinear
from models.mlp import MLP


class Generator(tf.keras.layers.Layer):
    """
    生成器
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        d_model=768,
        dropout=0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.d_model = d_model
        self.dropout = dropout
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.mapping_network = MappingNetwork(
            self.d_model,
            num_layers=8
        )

        # 输入位置编码
        self.patch_positional_embedding = PositionalEmbedding(
            sequence_length=self.num_patches,
            emb_dim=self.d_model,
        )

        self.generator_transformer_encoder = GeneratorEncoder(
            d_model,
            num_heads=8,
            num_layers=4,
            dropout=dropout,
        )
        self.sln1 = SLN(d_model)
        # 博里叶位置编码
        self.coordinates_positional_embedding = CoordinatesPositionalEmbedding(
            patch_size=patch_size,
            emb_dim=d_model,
            )

        self.siren = Siren(
            hidden_dim=d_model,
            hidden_layers=2,
            out_dim=num_channels,
            first_omega_0=30,
            hidden_omega_0=30,
            demodulation=True,
            outermost_linear=False
        )

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        w = self.mapping_network(x, training=training)
        # 输入位置编码
        x_pos = self.patch_positional_embedding()
        x = self.generator_transformer_encoder(x=x_pos, w=w, training=training)
        x = self.sln1(x, w, training=training)
        # 博里叶位置编码
        e_fou = self.coordinates_positional_embedding(x)
        x = self.siren((e_fou, x)) # (B*L, P*P, E)
        x = tf.reshape(x, [batch_size, self.grid_size, self.grid_size, self.patch_size, self.patch_size, self.num_channels])
        x = tf.transpose(x, perm=[0,1,3,2,4,5])
        x = tf.reshape(x, [batch_size, self.image_size, self.image_size, self.num_channels])
        return x


if __name__ == "__main__":
    layer = Generator(
        image_size=224,
        patch_size=16,
        num_channels=3,
        d_model=768
    )
    x = tf.random.uniform([2,1,768], dtype=tf.float32)
    o1 = layer(x, training=True)
    tf.print('o1:', tf.shape(o1))
    o1 = layer(x, training=False)
    tf.print('o1:', tf.shape(o1))