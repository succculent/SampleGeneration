from math import sqrt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Conv2D, MaxPool2D, Conv2DTranspose, Lambda, Reshape, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.python.keras.backend as K

# defining the conv1Dtranspose to use with keras Lambda because it does not exist in Keras
def Conv1DTranspose128(input_tensor):
    #x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=128, kernel_size=(3, 1), strides=(2, 1), padding='same')(input_tensor)
    #x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x
def Conv1DTranspose64(input_tensor):
    #x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=64, kernel_size=(3, 1), strides=(2, 1), padding='same')(input_tensor)
    #x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x
def Conv1DTranspose32(input_tensor):
    #x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=32, kernel_size=(3, 1), strides=(2, 1), padding='same')(input_tensor)
    #x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x
def Conv1DTranspose16(input_tensor):
    #x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=16, kernel_size=(3, 1), strides=(2, 1), padding='same')(input_tensor)
    #x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def create_encoder(input_length, encoding_dim):
    inputs = Input(shape=(None, input_length))
    conv1 = Conv1D(16, 3, padding="same", activation="relu")(inputs)
    conv1 = Conv1D(16, 3, padding="same", activation="relu")(conv1)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D()(norm1)
    
    conv2 = Conv1D(32, 3, padding="same", activation="relu")(pool1)
    conv2 = Conv1D(32, 3, padding="same", activation="relu")(conv2)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPool1D()(norm2)
    
    conv3 = Conv1D(64, 3, padding="same", activation="relu")(pool2)
    conv3 = Conv1D(64, 3, padding="same", activation="relu")(conv3)
    norm3 = BatchNormalization()(conv3)
    pool3 = MaxPool1D()(norm3)
    
    conv4 = Conv1D(128, 3, padding="same", activation="relu")(pool3)
    conv4 = Conv1D(128, 3, padding="same", activation="relu")(conv4)
    norm4 = BatchNormalization()(conv4)
    pool4 = MaxPool1D()(norm4)
    
    #conv5 = Conv1D(128, 1, padding="same", activation="relu")(pool4)
    
    # print(pool4.shape)
    # assert pool4.shape == (None, None, encoding_dim)

    encoder = Model(inputs, pool4)
    encoder.summary()
    
    return encoder

def create_decoder(encoding_dim):
    inputs = Input(shape=(None, None, encoding_dim))
    #replace below layers with lambda's of the conv1dtranspose lambda
    conv1 = Lambda(Conv1DTranspose128)(inputs)
    a1 = Activation("relu")(inputs)
    conv2 = Lambda(Conv1DTranspose64)(a1)
    a2 = Activation("relu")(conv2)
    conv3 = Lambda(Conv1DTranspose32)(a2)
    a3 = Activation("relu")(conv3)
    conv4 = Lambda(Conv1DTranspose16)(a3)
    a3 = Activation("relu")(conv4)
    #conv5 = Conv1D(3, 3, padding="same", activation="sigmoid")(conv4)
    
    decoder = Model(inputs, a3)
    decoder.summary()
    
    return decoder
