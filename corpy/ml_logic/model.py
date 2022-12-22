from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import tensorflow as tf
import numpy as np


def create_encoder(image_size, latent_dim):
    inputs = Input(shape=(image_size, image_size, 3))

    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    pre_flatten_shape = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(latent_dim)(x)

    encoder = Model(inputs, x, name="encoder")

    return encoder, pre_flatten_shape


def create_decoder(pre_flatten_shape, latent_dim):
    inputs = Input(shape=(latent_dim,))

    x = Dense(np.prod(pre_flatten_shape[1:]))(inputs)
    x = Reshape((pre_flatten_shape[1], pre_flatten_shape[2], pre_flatten_shape[3]))(x)
    x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)

    decoder = Model(inputs, x, name="decoder")

    return decoder


class AutoEncoder(Model):
    def __init__(self, encoder, decoder, * args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = metrics.Mean(name="loss")
        self.r_loss_tracker = metrics.Mean(name="reconstruction_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction = self.decoder(self.encoder(data))
            reconstruction_loss = tf.keras.losses.mean_squared_error(data, reconstruction)
            loss = reconstruction_loss

        gradients = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.r_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.r_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.r_loss_tracker
        ]

    def call(self, input):
        reconstruction = self.decoder(self.encoder(input))
        return reconstruction
