import os
from Embeddings.spectrogram import embed1d, embed2d
from Models.edModel import create_encoder, create_decoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

#read in files
path = os.getcwd() + "/Data/"

filenames = []
for filename in os.listdir(path):
    filenames.append(path+filename)

data, sr = embed1d(filenames)

# for item in data:
# 	print(item.shape)

encoding_dim = 256

encoder = create_encoder(len(data[0]), encoding_dim)
decoder = create_decoder(encoding_dim)

model = Model(encoder, decoder)
model.summary()
