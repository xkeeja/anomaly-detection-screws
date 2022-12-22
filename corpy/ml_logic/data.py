import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image_dataset_from_directory

# convert image dtypes to float
def to_float(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

# rotate images 10 degrees clockwise
def transform_rotate_cw(img):
    return tfa.image.rotate(
        img,
        -tf.constant(np.pi/18),
        fill_mode="reflect",
        interpolation="bilinear"
        )

# rotate images 10 degrees counterclockwise
def transform_rotate_ccw(img):
    return tfa.image.rotate(
        img,
        tf.constant(np.pi/18),
        fill_mode="reflect",
        interpolation="bilinear"
        )

# flip & rotate images 10 degrees
def transform_flip_rotate(img):
    return tfa.image.rotate(
        tf.image.flip_left_right(
            tf.image.flip_up_down(img)
            ),
        tf.constant(np.pi/18),
        fill_mode="reflect",
        interpolation="bilinear"
        )


# load data (& perform data augmentation on training set)
def load_data(batch_size, image_size, data_type):
    types = {
        'train': 'train/good',
        'test': 'test',
        'anomaly_only': 'train/not-good'
    }

    dim = (image_size, image_size)

    ds = image_dataset_from_directory(
        f'data/archive/{types[data_type]}',
        labels=None,
        image_size=dim,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
    )

    if data_type == 'train':
        ds = ds.concatenate(ds.map(transform_rotate_cw)).concatenate(ds.map(transform_rotate_ccw)).concatenate(ds.map(transform_flip_rotate))

    ds = (
        ds
        .map(to_float)
        .cache()
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return ds
