import os
from glob import glob
from Embeddings.spectrogram import embed1d, embed2d, embed1dlist
from Models.edModel import create_encoder, create_decoder, keras_custom_loss_function
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from soundfile import write

#read in files
path = os.getcwd() + "/Data/"

filenames = []
for filename in os.listdir(path):
    filenames.append(path+filename)

data, sr = embed1d(filenames)

encoding_dim = 256

encoder = create_encoder(data.shape[1], encoding_dim)
decoder = create_decoder(data.shape[1], encoding_dim)

#creating combined model
inputs = Input(shape=(data.shape[1], 1))
encoded = encoder(inputs)
decoded = decoder(encoded)
model = Model(inputs, decoded)

model.compile(optimizer="adam", loss="mse")

checkpointer = ModelCheckpoint("Models/ae/epoch{epoch}_loss{val_loss:.4f}.h5", save_best_only=True, verbose=1)
model.fit(data, data, epochs=500, validation_split=0.2, callbacks=[checkpointer])

saved_models = sorted(glob("models/ae/*"), key=os.path.getmtime)
autoencoder = load_model(saved_models[-1])
encoder = autoencoder.get_layer(index=1)
decoder = autoencoder.get_layer(index=2)

meTest = data[:5]
preds = autoencoder.predict(meTest)

for i in range(0, preds.shape[0]):
	write("FFToutput\\" + str(i) + "clap.wav", preds[i], sr[i])