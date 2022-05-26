import tensorflow as tf

def read_image(path, channels=3, normalized=True):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels)
    if normalized:
        img = tf.cast(img, dtype=tf.float32)
        img = (img-128.0)/128.0
    return img

def save_image(img, path, normalized=True):
    if normalized:
        img = img * 128.0 + 128.0
        img = tf.cast(img, dtype=tf.uint8)
    img = tf.image.encode_jpeg(img)
    tf.io.write_file(path, img)

def resize_image(img, target_height, target_width):
    img = tf.image.resize_with_pad(
        image=img,
        target_height=target_height,
        target_width=target_width,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )
    return img


if __name__ == '__main__':
    # img = read_image('Z:/Labels/lfw/lfw_mtcnnpy_182/Ralph_Lauren/Ralph_Lauren_0002.png')
    img = read_image('./test/test.jpg')
    img = resize_image(img,224,224)
    tf.print('img:', tf.shape(img), tf.math.reduce_min(img), tf.math.reduce_max(img))
    save_image(img, './test/test4.jpg')