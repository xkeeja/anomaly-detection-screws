import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import image_dataset_from_directory

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


def load_data(batch_size, image_size):
    dim = (image_size, image_size)

    # load train images, split train/val, data augmentation
    print(f'Loading train images...')
    ds_train, ds_val = image_dataset_from_directory(
        f'data/archive/train/good',
        labels=None,
        image_size=dim,
        interpolation='nearest',
        batch_size=batch_size,
        validation_split=0.2,
        subset='both',
        shuffle=True,
        seed=0
    )

    print('Data augmenting training images...')
    ds_train = ds_train.concatenate(ds_train.map(transform_rotate_cw)).concatenate(ds_train.map(transform_rotate_ccw)).concatenate(ds_train.map(transform_flip_rotate))
    print(f'{sum([len(batch) for batch in list(ds_train)])} total train images.')

    ds_val = ds_val.concatenate(ds_val.map(transform_rotate_cw)).concatenate(ds_val.map(transform_rotate_ccw)).concatenate(ds_val.map(transform_flip_rotate))
    print(f'{sum([len(batch) for batch in list(ds_val)])} total validation images.')

    # load test images
    print('Loading test images...')
    ds_test = image_dataset_from_directory(
        f'data/archive/test',
        labels=None,
        image_size=dim,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
    )

    # load anomaly images (for model reconstruction)
    print('Loading anomaly images...')
    ds_anom = image_dataset_from_directory(
        f'data/archive/train/not-good',
        labels=None,
        image_size=dim,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
    )


    ds_train = ds_train.map(to_float).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(to_float).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(to_float).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds_anom = ds_anom.map(to_float).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_anom
