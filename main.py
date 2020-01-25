import os
from Embeddings.wave import getEmbedding
from Models import LSTM

#read in files
path = os.getcwd() + "\\Data\\"

filenames = []
for filename in os.listdir(path):
    filenames.append(path+filename)

embeddings = getEmbedding(filenames)