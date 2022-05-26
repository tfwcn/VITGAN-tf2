import tensorflow as tf
import sys

sys.path.append('')

from models.isn import ISN


class PatchEmbedding(tf.keras.layers.Layer):
    """
    2D Image to Patch Embedding
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        overlapping=3, # 图像重叠部分
        emb_dim=768,
        discriminator=True,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.overlapping = overlapping
        self.emb_dim = emb_dim
        self.discriminator = discriminator
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = tf.keras.layers.Dense(emb_dim, use_bias=False)
        # if discriminator:
        #     self.proj = ISN(self.proj)

        self.create_indexes()

    def create_indexes(self):
        '''创建切片下标'''
        self.all_indexes = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                now_y_start = y * self.patch_size
                now_y_end = (y+1) * self.patch_size
                now_x_start = x * self.patch_size
                now_x_end = (x+1) * self.patch_size
                # 加重叠部分，边缘则加到同一边
                if y == 0:
                    now_y_end += 2 * self.overlapping
                elif y == self.grid_size-1:
                    now_y_start -= 2 * self.overlapping
                else:
                    now_y_start -= self.overlapping
                    now_y_end += self.overlapping
                if x == 0:
                    now_x_end += 2 * self.overlapping
                elif x == self.grid_size-1:
                    now_x_start -= 2 * self.overlapping
                else:
                    now_x_start -= self.overlapping
                    now_x_end += self.overlapping
                self.all_indexes.append(
                    (now_y_start, now_y_end, now_x_start, now_x_end)
                )
        # print('all_indexes:', self.all_indexes)

    def call(self, x):
        batch = tf.shape(x)[0]
        patch_list = []
        for now_y_start, now_y_end, now_x_start, now_x_end in self.all_indexes:
            patch_x = x[:,now_y_start:now_y_end,now_x_start:now_x_end,:]
            patch_list.append(tf.reshape(patch_x,[batch,1,-1]))
        x = tf.concat(patch_list, axis=1)
        x = self.proj(x)
        return x


if __name__ == "__main__":
    layer = PatchEmbedding(
        image_size=224,
        patch_size=16,
        overlapping=3,
        embed_dim=768
    )
    # x = tf.random.uniform([2,224,224,3], dtype=tf.float32)
    x = tf.io.read_file('./test.jpg')
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.expand_dims(x, axis=0)
    x = tf.image.crop_and_resize(x, [[0,0,1,1]], [0], [224,224])
    o1 = layer(x)
    tf.print('o1:', tf.shape(o1))
    o1 = tf.reshape(o1, [1, 14, 14, 22, 22, 3])
    o1 = tf.transpose(o1, perm=[0,1,3,2,4,5])
    o1 = tf.reshape(o1, [1, 308, 308, 3])
    o1 = tf.image.encode_jpeg(tf.cast(o1[0], dtype=tf.uint8))
    tf.io.write_file('./test3.jpg', o1)