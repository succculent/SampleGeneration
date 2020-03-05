from math import sqrt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Conv2D, MaxPool2D, Conv2DTranspose, Lambda, Reshape, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.python.keras.backend as K

def create_encoder(input_length, encoding_dim):
    inputs = Input(shape=(None, input_length))
    conv1 = Conv1D(2048, 3, padding="same", activation="relu")(inputs)
    conv1 = Conv1D(2048, 3, padding="same", activation="relu")(conv1)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D()(norm1)
    
    conv2 = Conv1D(1024, 3, padding="same", activation="relu")(pool1)
    conv2 = Conv1D(1024, 3, padding="same", activation="relu")(conv2)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPool1D()(norm2)
    
    conv3 = Conv1D(512, 3, padding="same", activation="relu")(pool2)
    conv3 = Conv1D(512, 3, padding="same", activation="relu")(conv3)
    norm3 = BatchNormalization()(conv3)
    pool3 = MaxPool1D()(norm3)
    
    conv4 = Conv1D(256, 3, padding="same", activation="relu")(pool3)
    conv4 = Conv1D(256, 3, padding="same", activation="relu")(conv4)
    norm4 = BatchNormalization()(conv4)
    pool4 = MaxPool1D()(norm4)
    
    conv5 = Conv1D(256, 1, padding="same", activation="relu")(pool4)
    
    # print(pool4.shape)
    # assert pool4.shape == (None, None, encoding_dim)

    encoder = Model(inputs, conv5)
    # encoder.summary()
    
    return encoder

def create_decoder(input_length, encoding_dim):
    inputs = Input(shape=(None, encoding_dim))
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(inputs)
    x = Conv2DTranspose(filters=256, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=512, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=1024, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=2048, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=input_length, kernel_size=(3, 1), strides=(2, 1), padding='same')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    x = Activation("sigmoid")(x)
    decoder = Model(inputs, x)
    # decoder.summary()
    
    return decoder