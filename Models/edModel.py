from math import sqrt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Conv2D, MaxPool2D, Conv2DTranspose, Lambda, Reshape, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.python.keras.backend as K
import tensorflow.keras.backend as kb
import numpy as np
import librosa

def create_encoder(input_length, encoding_dim):
    inputs = Input(shape=(input_length, 1))
    conv1 = Conv1D(16, 3, padding="same", activation="relu")(inputs)
    conv1 = Conv1D(16, 3, padding="same", activation="relu")(conv1)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D(data_format='channels_last')(norm1)
    
    conv2 = Conv1D(32, 3, padding="same", activation="relu")(pool1)
    conv2 = Conv1D(32, 3, padding="same", activation="relu")(conv2)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPool1D(data_format='channels_last')(norm2)
    
    conv3 = Conv1D(64, 3, padding="same", activation="relu")(pool2)
    conv3 = Conv1D(64, 3, padding="same", activation="relu")(conv3)
    norm3 = BatchNormalization()(conv3)
    pool3 = MaxPool1D(data_format='channels_last')(norm3)
    
    conv4 = Conv1D(256, 3, padding="same", activation="relu")(pool3)
    conv4 = Conv1D(256, 3, padding="same", activation="relu")(conv4)
    norm4 = BatchNormalization()(conv4)
    pool4 = MaxPool1D(data_format='channels_last')(norm4)
    
    conv5 = Conv1D(256, 1, padding="same", activation="relu")(pool4)

    encoder = Model(inputs, conv5)
    
    return encoder

def create_decoder(input_length, encoding_dim):
    inputs = Input(shape=(input_length//16, encoding_dim))
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(inputs)
    x = Conv2DTranspose(filters=256, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=64, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=32, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=16, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Conv1D(1, 16, padding = "same", activation = "sigmoid")(x)

    decoder = Model(inputs, x)
    
    return decoder

def keras_custom_loss_function(real, pred):
    custom_loss_value = kb.mean(kb.sum(kb.square(np.fft.fft(real)-np.fft.fft(pred))))
    return custom_loss_value