import os
import tensorflow.keras
from tensorflow.keras import Sequential, Model
import Embeddings.waveEmbedding
import Models.lstmModel

#read in files
path = os.getcwd() + "\\Data\\"

filenames = []
for filename in os.listdir(path):
    filenames.append(path+filename)

embeddings = Embeddings.waveEmbedding.getEmbedding(filenames)

length = len(embeddings)

model =Models.lstmModel.getModel(length)
model.desc()