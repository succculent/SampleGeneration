import os
from Embeddings.spectrogram import embed1d, embed2d
from Models.edModel import create_encoder, create_decoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

#read in files
path = os.getcwd() + "/Data/"

filenames = []
for filename in os.listdir(path):
    filenames.append(path+filename)

data, sr = embed1d(filenames)

# data = np.expand_dims(data, axis=0)
print(data.shape)
# for item in data:
# 	print(item.shape)

encoding_dim = 256

encoder = create_encoder(data.shape[2], encoding_dim)
decoder = create_decoder(data.shape[2], encoding_dim)

inputs = Input(shape=(data.shape[1], data.shape[2]))
encoded = encoder(inputs)
decoded = decoder(encoded)

model = Model(inputs, decoded)
model.summary()

# model.compile(optimizer="adam", loss="mse")
# model.fit(data, data, epochs=50, validation_split=0.2)