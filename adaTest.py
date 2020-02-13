import os
from tensorflow import 
import Embeddings.librosa_chroma_stft

#read in files
path = os.getcwd() + "\\Data\\"

filenames = []
for filename in os.listdir(path):
    filenames.append(path+filename)

data, label, maxLen = Embeddings.librosa_chroma_stft.getEmbedding(filenames)

print(data.shape)
print(len(label))


#print(maxLen.shape)



def cnn():
  inputs = Input(shape=(30, 12, 22))

  conv1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
  conv2 = Conv2D(32, 3, activation="relu", padding="same")(conv1)
  pool1 = MaxPool2D()(conv2)
  conv3 = Conv2D(64, 3, activation="relu", padding="same")(pool1)
  conv4 = Conv2D(64, 3, activation="relu", padding="same")(conv3)
  pool2 = MaxPool2D()(conv4)

  flatten = Flatten()(pool2)
  dense1 = Dense(64, activation="relu")(flatten)
  logits = Dense(maxLen, activation="softmax")(dense1)

  model = Model(inputs, logits)
  model.summary()
  return model

model = cnn()