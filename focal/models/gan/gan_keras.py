import datetime
import time

import numpy as np
import pandas as pd
import tensorflow as tf
#from scipy.ndimage import center_of_mass
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from evaluation_utils import find_radius_circle_bisection, generate_masks
from sklearn.metrics import mean_squared_error


def prepare_generator(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    x_cond = tf.keras.layers.Dense(8, use_bias=False)(input_cond)
    x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    x_cond = tf.keras.layers.Dropout(0.3)(x_cond)
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, x_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Reshape((7, 7, 256))(x1)

    x1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2D(filters=1, kernel_size=8, strides=(1, 1), activation='relu')(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')


def prepare_generator_upsampling(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    x_cond = tf.keras.layers.Dense(8, use_bias=False)(input_cond)
    x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    x_cond = tf.keras.layers.Dropout(0.3)(x_cond)
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, x_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x1 = tf.keras.layers.Reshape((7, 7, 256))(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(3, 3), interpolation="nearest")(x1)  # 7→21
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(5, 5), interpolation="nearest")(x1)  # 21→105
    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    # Final output layer
    x1 = tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="relu")(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')


def res_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, (3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def prepare_generator_upsampling_bigger(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))

    x_cond = tf.keras.layers.Dense(8, use_bias=False)(input_cond)
    x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    x_cond = tf.keras.layers.Dropout(0.3)(x_cond)
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, x_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Reshape((7, 7, 256))(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(3, 3), interpolation="nearest")(x1)  # 7→21
    x1 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(5, 5), interpolation="nearest")(x1)  # 21→105
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 32)
    x1 = res_block(x1, 32)
    x1 = res_block(x1, 32)

    # Final output layer
    x1 = tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="relu")(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')


def prepare_generator_upsampling_medium(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    x_cond = tf.keras.layers.Dense(8, use_bias=False)(input_cond)
    x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    x_cond = tf.keras.layers.Dropout(0.3)(x_cond)
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, x_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Reshape((7, 7, 256))(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 7→14
    x1 = tf.keras.layers.Conv2D(256, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 14->28
    x1 = tf.keras.layers.Conv2D(128, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 128)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 28→56
    x1 = tf.keras.layers.Conv2D(128, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 128)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 56→112
    x1 = tf.keras.layers.Conv2D(64, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)


    x1 = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(x1) # 112->110
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 32)

    x1 = tf.keras.layers.Conv2D(32, (5, 5), use_bias=False)(x1) # 110->106
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 32)
    # x1 = res_block(x1, 32)
    # x1 = res_block(x1, 32)

    # Final output layer
    x1 = tf.keras.layers.Conv2D(1, (2, 2), activation="relu")(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')


def prepare_generator_upsampling_smaller(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    x_cond = tf.keras.layers.Dense(8, use_bias=False)(input_cond)
    x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    x_cond = tf.keras.layers.Dropout(0.3)(x_cond)
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, x_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Reshape((7, 7, 256))(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 7→14
    x1 = tf.keras.layers.Conv2D(256, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 14->28
    x1 = tf.keras.layers.Conv2D(128, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 128)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 28→56
    x1 = tf.keras.layers.Conv2D(64, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 64)

    x1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x1)  # 56→112
    x1 = tf.keras.layers.Conv2D(64, (2, 2), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)


    x1 = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(x1) # 112->110
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 32)

    x1 = tf.keras.layers.Conv2D(32, (5, 5), use_bias=False)(x1) # 110->106
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block(x1, 32)

    # Final output layer
    x1 = tf.keras.layers.Conv2D(1, (2, 2), activation="relu")(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')



def res_block_old(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, (3,3), padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters, (3,3), padding="same")(x)
    return tf.keras.layers.Add()([shortcut, x])


def prepare_generator_upsampling_bigger_old(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    x_cond = tf.keras.layers.Dense(8, use_bias=False)(input_cond)
    x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    x_cond = tf.keras.layers.Dropout(0.3)(x_cond)
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, x_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x1 = tf.keras.layers.Reshape((7, 7, 256))(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(3, 3), interpolation="nearest")(x1)  # 7→21
    x1 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.UpSampling2D(size=(5, 5), interpolation="nearest")(x1)  # 21→105
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = res_block_old(x1, 128)

    # Final output layer
    x1 = tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="relu")(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')



def prepare_generator_nocondlayers(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, input_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Reshape((7, 7, 256))(x1)

    x1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2D(filters=1, kernel_size=8, strides=(1, 1), activation='relu')(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')


def prepare_generator_smaller(latent_dim, cond_dim):
    input_img = tf.keras.layers.Input(shape=(latent_dim,))
    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    inputs = tf.keras.layers.Concatenate(axis=1)([input_img, input_cond])

    x1 = tf.keras.layers.Dense(7 * 7 * 128, use_bias=False)(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Reshape((7, 7, 128))(x1)

    x1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv2D(filters=1, kernel_size=8, strides=(1, 1), activation='relu')(x1)

    return tf.keras.Model([input_img, input_cond], x1, name='generator')


def prepare_discriminator(img_shape, cond_dim):
    input_img = tf.keras.layers.Input(shape=img_shape)

    x = tf.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same')(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    x_cond = tf.keras.layers.Dense(16)(input_cond)
    x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    x_cond = tf.keras.layers.Dropout(0.3)(x_cond)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_cond])
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(1)(x)  # wartości < 0: fałszywy obraz, wpp - prawdziwy

    return tf.keras.Model([input_img, input_cond], x, name='discriminator')


def prepare_discriminator_smaller(img_shape, cond_dim):
    input_img = tf.keras.layers.Input(shape=img_shape)

    x = tf.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same')(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(64)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    # x = tf.keras.layers.Dropout(0.3)(x)

    input_cond = tf.keras.layers.Input(shape=(cond_dim,))
    # x_cond = tf.keras.layers.Dense(16)(input_cond)
    # x_cond = tf.keras.layers.BatchNormalization()(x_cond)
    # x_cond = tf.keras.layers.LeakyReLU()(x_cond)
    # x_cond = tf.keras.layers.Dropout(0.3)(x_cond)

    x = tf.keras.layers.Concatenate(axis=1)([x, input_cond])
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(1)(x)  # wartości < 0: fałszywy obraz, wpp - prawdziwy

    return tf.keras.Model([input_img, input_cond], x, name='discriminator')


def prepare_datasets(particles_fnm, responses_fnm, batch_size=256, val_size=0.1, test_size=0.2, return_scaler=True):
    particles = pd.read_csv(particles_fnm, index_col=0)
    responses = np.load(responses_fnm)
    responses_log = np.log(responses + 1)
    particles_train, particles_test, responses_train, responses_test = train_test_split(particles, responses_log,
                                                                                        test_size=test_size,
                                                                                        random_state=42,
                                                                                        shuffle=True)
    particles_train, particles_val, responses_train, responses_val = train_test_split(particles_train, responses_train,
                                                                                      test_size=val_size / (
                                                                                              1 - test_size),
                                                                                      random_state=43,
                                                                                      shuffle=True)

    particles_scaler = MinMaxScaler()
    particles_train_scaled = particles_scaler.fit_transform(particles_train)
    particles_val_scaled = particles_scaler.transform(particles_val)
    particles_test_scaled = particles_scaler.transform(particles_test)

    train_dataset_with_cond = prepare_dataset_with_cond(responses_train, particles_train_scaled, particles_train, batch_size, prepare_fake_cond=True, reshuffle_each_iteration=True)
    val_dataset_with_cond = prepare_dataset_with_cond(responses_val, particles_val_scaled, particles_val, batch_size, prepare_fake_cond=False, reshuffle_each_iteration=False)
    test_dataset_with_cond = prepare_dataset_with_cond(responses_test, particles_test_scaled, particles_test, batch_size, prepare_fake_cond=False, reshuffle_each_iteration=False)

    if return_scaler:
        return train_dataset_with_cond, val_dataset_with_cond, test_dataset_with_cond, particles_scaler
    return train_dataset_with_cond, val_dataset_with_cond, test_dataset_with_cond


def prepare_dataset_with_cond(responses, particles, particles_not_scaled, batch_size, prepare_fake_cond=False, reshuffle_each_iteration=True):
    dataset_size = responses.shape[0]
    x_dataset = tf.data.Dataset.from_tensor_slices(responses).batch(batch_size)
    cond_dataset = tf.data.Dataset.from_tensor_slices(particles).batch(batch_size)
    if prepare_fake_cond:
        fake_cond_dataset = tf.data.Dataset.from_tensor_slices(particles).shuffle(dataset_size, seed=42).batch(batch_size)
        fake_cond_dataset_not_scaled = tf.data.Dataset.from_tensor_slices(particles_not_scaled).shuffle(dataset_size, seed=42).batch(batch_size)
        return tf.data.Dataset.zip((x_dataset, cond_dataset, fake_cond_dataset, fake_cond_dataset_not_scaled)).shuffle(dataset_size, reshuffle_each_iteration=reshuffle_each_iteration)
    return tf.data.Dataset.zip((x_dataset, cond_dataset)).shuffle(dataset_size, reshuffle_each_iteration=reshuffle_each_iteration)


def discriminator_loss(real_output, fake_output, history_objects):
    # loss using label smoothing
    real_loss = cross_entropy(tf.ones_like(real_output) + tf.random.uniform(shape=real_output.shape, minval=-0.15, maxval=0), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, minval=0, maxval=0.1), fake_output)
    total_loss = real_loss + fake_loss

    d_acc_r, d_acc_f = history_objects
    d_acc_r.update_state(tf.ones_like(real_output), tf.math.sigmoid(real_output))
    d_acc_f.update_state(tf.zeros_like(fake_output), tf.math.sigmoid(fake_output))
    return total_loss


def generator_loss(fake_output, history_objects):
    g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    g_acc = history_objects
    g_acc.update_state(tf.ones_like(fake_output), tf.math.sigmoid(fake_output))
    return g_loss


def compute_expected_hit_pos(eta, phi, z=7.17):
    common = z * tf.math.tan(2 * tf.math.atan(tf.math.exp(eta)))
    x = common * tf.math.cos(phi)
    y = common * tf.math.sin(phi)
    return tf.stack([x, y], axis=1)


def pos_to_coords(pos):
    cell_size = 6.55 / 7 * 0.01
    x, y = pos[..., 0], pos[..., 1]
    x_coord = x / cell_size
    y_coord = y / cell_size
    x_coord = x_coord * (-1) + 52
    y_coord = y_coord * (-1) + 52
    return tf.cast(tf.stack([x_coord, y_coord], axis=1), dtype=tf.float32)


def physical_loss(detector_response, eta, phi, history_objects, loss_type):
    xy = compute_expected_hit_pos(eta, phi)
    recomputed_coords = pos_to_coords(xy)
    reshaped_response = tf.reshape(detector_response, tf.shape(detector_response)[:3])
    com = center_of_mass_tf(reshaped_response)

    #loss = mse(recomputed_coords, com)
    loss = L1_loss(recomputed_coords, com) if loss_type == "L1" else L2_loss(recomputed_coords, com)

    ph_metric = history_objects
    ph_metric.update_state(recomputed_coords, com)

    return loss


def center_of_mass_tf_2d(image):
    height, width = tf.shape(image)[0], tf.shape(image)[1]

    y_coords = tf.range(height, dtype=tf.float32)
    x_coords = tf.range(width, dtype=tf.float32)
    y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing="ij")

    total_mass = tf.reduce_sum(image)
    total_mass = tf.maximum(total_mass, tf.constant(1e-8, dtype=tf.float32))

    y_com = tf.reduce_sum(image * y_grid) / total_mass
    x_com = tf.reduce_sum(image * x_grid) / total_mass

    #return y_com, x_com
    return tf.stack([y_com, x_com])


def center_of_mass_tf(images):
    batch_size, height, width = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2]

    y_coords = tf.range(height, dtype=tf.float32)
    x_coords = tf.range(width, dtype=tf.float32)
    y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing="ij")

    # Expand grids for broadcasting across the batch
    y_grid = tf.expand_dims(y_grid, axis=0)
    x_grid = tf.expand_dims(x_grid, axis=0)

    # Compute total mass for each image
    total_mass = tf.reduce_sum(images, axis=[1, 2], keepdims=True)  # Shape: (N, 1, 1)

    # Avoid division by zero
    total_mass = tf.maximum(total_mass, tf.constant(1e-8, dtype=tf.float32))

    # Compute weighted sums of coordinates
    y_com = tf.reduce_sum(images * y_grid, axis=[1, 2]) / tf.squeeze(total_mass, axis=[1, 2])  # Shape: (N,)
    x_com = tf.reduce_sum(images * x_grid, axis=[1, 2]) / tf.squeeze(total_mass, axis=[1, 2])  # Shape: (N,)

    # Return the centers of mass as (N, 2)
    return tf.stack([y_com, x_com], axis=1)



@tf.function
def train_step(generator, discriminator, data, noise_dim, history_objects, ph_loss_type, ph_loss_scaler=0.005):
    images, cond, noise_cond, noise_cond_not_scaled = data
    g_acc, d_acc_r, d_acc_f, ph_metric = history_objects
    batch_size = tf.shape(images)[0]

    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, noise_cond], training=True)

        real_output = discriminator([images, cond], training=True)
        fake_output = discriminator([generated_images, noise_cond], training=True)

        gen_loss = generator_loss(fake_output, g_acc)
        disc_loss = discriminator_loss(real_output, fake_output, (d_acc_r, d_acc_f))

        if ph_metric is not None:
            ph_loss = physical_loss(generated_images, noise_cond_not_scaled[:, 1], noise_cond_not_scaled[:, 2], ph_metric, ph_loss_type)
            total_gen_loss = gen_loss + ph_loss_scaler * ph_loss
        else:
            total_gen_loss = gen_loss

    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    if ph_metric is not None:
        return gen_loss, disc_loss, ph_loss
    else:
        return gen_loss, disc_loss, 0.0


def train(models, datasets, epochs, noise_dim, history_objects, timestamp, save_dir, ph_loss_type, ph_loss_scaler):
    history = []
    g_acc, d_acc_r, d_acc_f, ph_metric = history_objects
    generator, discriminator = models
    train_dataset, val_dataset = datasets
    generator_fnm = f'generator_{timestamp}'
    discriminator_fnm = f'discriminator_{timestamp}'
    print(f"Generator fnm: {generator_fnm}\ndiscriminator fnm: {discriminator_fnm}\n", flush=True)
    #best_val_mse = float('inf')
    best_val_shower_difference = float('inf')

    img_size = 105
    center = (105, 105)
    diam_to_indices = generate_masks(center, img_size)
    radi_org = {0.25: [], 0.9: []}
    center_org = []
    for pc in radi_org:
        for image_batch in val_dataset:
            imgs, _ = image_batch
            imgs = imgs.numpy()
            for img in imgs:
                radius = find_radius_circle_bisection(img, diam_to_indices, fraction=pc)
                radi_org[pc].append(radius)

    for image_batch in val_dataset:
        imgs, _ = image_batch
        imgs = imgs.numpy()
        for img in imgs:
            com = np.argwhere(img == img.max())[0]
            center_org.append(com[:2])
    center_org = np.array(center_org)

    for epoch in range(epochs):
        start = time.time()

        for image_batch in train_dataset:
            gen_loss, disc_loss, ph_loss = train_step(generator, discriminator, image_batch, noise_dim, history_objects, ph_loss_type, ph_loss_scaler)
            history.append([gen_loss, disc_loss, ph_loss,
                            100 * d_acc_r.result().numpy(),
                            100 * d_acc_f.result().numpy(),
                            100 * g_acc.result().numpy(),
                            2 * ph_metric.result().numpy() if ph_metric is not None else None,
                            ])

        if epoch % 10 == 0:
            print("%d [D real acc: %.2f%%] [D fake acc: %.2f%%] [G acc: %.2f%%] [ph err: %.2f]" % (
                epoch,
                100 * d_acc_r.result().numpy(),
                100 * d_acc_f.result().numpy(),
                100 * g_acc.result().numpy(),
                2 * ph_metric.result().numpy() if ph_metric is not None else 0.0), flush=True)
        if epoch % 50 == 0:
            save_sample_images(generator, val_dataset, latent_dim, f"{timestamp}_epoch{epoch}", save_dir)

        radi_val = {0.25: [], 0.9: []}
        center_val = []
        for image_batch in val_dataset:
            img, cond = image_batch
            z = tf.random.normal([img.shape[0], noise_dim])
            img_gen = generator.predict([z, cond], verbose=0)
            curr_radi = compute_radi_gen(diam_to_indices, radi_org, img_gen)
            for k in radi_val:
                radi_val[k].extend(curr_radi[k])
            for img in img_gen:
                com = np.argwhere(img == img.max())[0]
                center_val.append(com[:2])
        center_val = np.array(center_val)
        val_radi_difference = compare_radi(radi_org, radi_val)
        val_center_difference = compare_center_pos(center_org, center_val)
        if epoch % 10 == 0:
            print(f"Current mean val radi difference: {val_radi_difference}")
            print(f"Current mean val position difference: {val_center_difference}")
        if val_radi_difference + val_center_difference < best_val_shower_difference:
            generator.save(save_dir + '/' + f"{generator_fnm}.keras")
            discriminator.save(save_dir + '/' + f"{discriminator_fnm}.keras")
            best_val_shower_difference = val_radi_difference + val_center_difference
            print(f"Shower stats model saved. Current sum of shower shape differences: {best_val_shower_difference}", flush=True)
            print(f"Current mean val radi difference: {val_radi_difference}")
            print(f"Current mean val position difference: {val_center_difference}")
        # val_mse = tf.keras.metrics.MeanSquaredError()
        # for image_batch in val_dataset:
        #     img, cond = image_batch
        #     z = tf.random.normal([img.shape[0], noise_dim])
        #     img_gen = generator.predict([z, cond], verbose=0)
        #     val_mse.update_state(img, img_gen)
        # val_mse_res = val_mse.result().numpy()
        # if val_mse_res < best_val_mse:
        #     generator.save(save_dir + '/' + f"{generator_fnm}.keras")
        #     discriminator.save(save_dir + '/' + f"{discriminator_fnm}.keras")
        #     best_val_mse = val_mse_res
        #     print(f"MSE model saved. Current MSE: {best_val_mse}", flush=True)

        if ph_metric is not None:
            ph_metric.reset_state()
        g_acc.reset_state()
        d_acc_r.reset_state()
        d_acc_f.reset_state()
        print(f'Time for epoch {epoch} is {time.time() - start} s', flush=True)

    generator.save(save_dir + '/' + f"{generator_fnm}_lastmodel.keras")
    discriminator.save(save_dir + '/' + f"{discriminator_fnm}_lastmodel.keras")
    np.save(save_dir + "/history_" + timestamp + ".npy", np.array(history))
    print(f"History file saved under history_{timestamp}.npy", flush=True)


def test_model(generator):
    test_mse_log = tf.keras.metrics.MeanSquaredError()
    test_mse = tf.keras.metrics.MeanSquaredError()
    for image_batch in test_dataset:
        img, cond = image_batch
        z = tf.random.normal([img.shape[0], latent_dim])
        img_gen = generator.predict([z, cond], verbose=0)
        test_mse_log.update_state(img, img_gen)
        img_org = tf.math.exp(img) - 1
        img_gen = tf.math.exp(img_gen) - 1
        test_mse.update_state(img_org, img_gen)
    test_mse_log_res = test_mse_log.result().numpy()
    test_mse_res = test_mse.result().numpy()
    print(f"Test MSE on log images: {test_mse_log_res}\nTest MSE on org images: {test_mse_res}", flush=True)


def save_sample_images(generator, test_dataset, noise_dim, timestamp, models_dir, num_examples_to_generate=16):
    for test_batch in test_dataset.take(1):
        test_sample_data = test_batch[0][0:num_examples_to_generate, :, :]
        test_sample_cond = test_batch[1][0:num_examples_to_generate, :]
        test_sample_noise = tf.random.normal([num_examples_to_generate, noise_dim])

    test_sample = [test_sample_noise, test_sample_cond]
    predictions = generator(test_sample, training=False)

    figsize = (8, 4)
    fig = plt.figure(figsize=figsize)

    for i in range(test_sample_data.shape[0]):
        plt.subplot(figsize[1], figsize[0], i + 1)
        plt.imshow(test_sample_data[i, :, :], interpolation='none', cmap='gnuplot')
        plt.axis('off')
    for i in range(predictions.shape[0]):
        plt.subplot(figsize[1], figsize[0], figsize[1] // 2 * figsize[0] + i + 1)
        plt.imshow(predictions[i, :, :, 0], interpolation='none', cmap='gnuplot')
        plt.axis('off')

    plt.savefig(models_dir + f'/generator_{timestamp}.png', bbox_inches='tight')


def compute_radi_gen(diam_to_indices, radi_org, generated_imgs):
    radi_gen = {k: [] for k in radi_org}
    for frac in radi_org:
        for idx in range(generated_imgs.shape[0]):
            radius = find_radius_circle_bisection(generated_imgs[idx], diam_to_indices, fraction=frac)
            radi_gen[frac].append(radius)
    return radi_gen


def compare_radi(radi_org, radi_gen):
    res = {}
    for k in radi_org:
        res[k] = mean_squared_error(radi_org[k], radi_gen[k])
    return np.mean(list(res.values()))


def compare_center_pos(center_org, center_gen):
    return np.mean((center_org[:, 0] - center_gen[:, 0]) ** 2 + (center_org[:, 1] - center_gen[:, 1]) ** 2)


def L2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1)) # Keeps per-sample sum


def L1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.math.abs(y_true - y_pred), axis=1)) # Keeps per-sample sum


class SumSquaredErrorMetric(tf.keras.metrics.Metric):
    def __init__(self, name="sum_squared_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        squared_errors = tf.reduce_sum(tf.square(y_true - y_pred), axis=1)  # Sum over features
        batch_mean = tf.reduce_mean(squared_errors)  # Mean over batch
        self.total_loss.assign_add(batch_mean)
        self.count.assign_add(1)

    def result(self):
        return self.total_loss / self.count  # Running mean over batches

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.count.assign(0.0)


if __name__ == "__main__":
    base_dir = '/net/pr2/projects/plgrid/plggdaisnet'
    particles_fnm = 'filtered_focal_particles_new.csv'
    responses_fnm = 'filtered_focal_responses_new.npy'
    epochs = 500
    batch_size = 512
    latent_dim = 32
    cond_dim = 3
    img_shape = (105, 105, 1)
    lr_generator = 5e-5 #1e-3  # 1e-4
    lr_discriminator = 1e-5 #1e-4  # 5e-5
    beta_1 = 0.5
    models_base_dir = base_dir + '/models/focal/gan'
    dataset_base_dir = base_dir + '/data/focal'
    add_ph_loss = True
    ph_loss_scaler = 0.0001
    ph_loss_type = "L2"
    generator_func = prepare_generator_upsampling_smaller
    discriminator_func = prepare_discriminator
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S:%f")

    print(f"Datasets: {particles_fnm}, {responses_fnm}", flush=True)
    print(f"PARAMS:\nepochs: {epochs}\nbatch size: {batch_size}\nlatent dim: {latent_dim}\ncond dim: {cond_dim}\n"
          f"lr generator: {lr_generator}\nlr discriminator: {lr_discriminator}\nbeta_1: {beta_1}\n"
          f"ph loss mode: {add_ph_loss}\nph loss scaler: {ph_loss_scaler}; scale only ph loss\n"
          f"ph loss type: {ph_loss_type}\ngenerator func: {generator_func.__name__}\ndiscriminator func: {discriminator_func.__name__}\n"
          f"timestamp: {timestamp}\n", flush=True)

    train_dataset, val_dataset, test_dataset, scaler = prepare_datasets(dataset_base_dir + '/' + particles_fnm,
                                                                        dataset_base_dir + '/' + responses_fnm,
                                                                        batch_size=batch_size, val_size=0.1,
                                                                        test_size=0.2, return_scaler=True)
    generator = generator_func(latent_dim, cond_dim)
    generator.summary()
    discriminator = discriminator_func(img_shape, cond_dim)
    discriminator.summary()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    d_acc_r = tf.keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    d_acc_f = tf.keras.metrics.BinaryAccuracy(name="d_acc_f", threshold=0.5)
    g_acc = tf.keras.metrics.BinaryAccuracy(name="g_acc_g", threshold=0.5)
    ph_metric = tf.keras.metrics.MeanSquaredError(name="ph_mse") if ph_loss_type == "L2" \
        else tf.keras.metrics.MeanAbsoluteError(name="ph_mae") if ph_loss_type == "L1" \
        else None
    #ph_metric = tf.keras.metrics.MeanSquaredError() if add_ph_loss else None

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_generator, beta_1=beta_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_discriminator, beta_1=beta_1)

    train((generator, discriminator), (train_dataset, val_dataset), epochs, latent_dim,
          (g_acc, d_acc_r, d_acc_f, ph_metric), timestamp, models_base_dir, ph_loss_type, ph_loss_scaler)
    save_sample_images(generator, test_dataset, latent_dim, f"{timestamp}_lastmodel", models_base_dir)

    ### test ###
    saved_generator = tf.keras.models.load_model(models_base_dir + f'/generator_{timestamp}.keras')
    test_model(saved_generator)
    save_sample_images(saved_generator, test_dataset, latent_dim, timestamp, models_base_dir)
