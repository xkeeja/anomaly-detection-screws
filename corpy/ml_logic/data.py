import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image_dataset_from_directory


def convert_to_float(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def trans1(img):
    return tfa.image.rotate(tf.image.flip_left_right(tf.image.flip_up_down(img)),-.2,fill_mode="reflect",interpolation="bilinear")


def trans2(img):
    return tfa.image.rotate(img,-.2,fill_mode="reflect",interpolation="bilinear")


def trans3(img):
    return tfa.image.rotate(img,.2,fill_mode="reflect",interpolation="bilinear")


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
        ds1, dst1, dst2, dst3 = ds, ds.map(trans1), ds.map(trans2), ds.map(trans3)

        ds = ds1.concatenate(dst1).concatenate(dst2).concatenate(dst3)

    ds = (
        ds
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return ds
