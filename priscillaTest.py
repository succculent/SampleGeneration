import os
import librosa
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, LSTM

#read in files
path = os.getcwd() + "/Data/"

filenames = []
for filename in os.listdir(path):
    filenames.append(path+filename)

#load audio time series of each sample into loaded
#loads the sample rate of each sample into sr
loaded = []
sr = []
for path in filenames:
	x , y = librosa.load(path)
	loaded.append(x)
	sr.append(y)

#finds the maximum sample length for generation
maxLen = 0
for x in loaded:
	if (len(x) > maxLen):
		maxLen = len(x)

'''
#pads audio time series to be the same length
for x in loaded:
	x = np.pad(x, (0, maxLen-len(x)))
'''

#extracts features from each audio time series and loads into extracted
#also finds the largest shape of extracted data
extracted = []
maxLena = 0
maxLenb = 0
for x in range(0, len(sr)):
	extracted.append(librosa.feature.chroma_stft(loaded[x], sr[x]))
	a, b = extracted[x].shape
	if (a > maxLena):
		maxLena = a
	if (b > maxLenb):
		maxLenb = b

padded = []
#pads extracted data
for x in extracted:
	a, b = x.shape
	x = np.pad(x, ((0,0), (maxLena-a,maxLenb-b)))
	padded.append(x)

label = loaded
data = np.dstack((padded))
data = np.rollaxis(data,-1)
data = np.expand_dims(data, axis=0)
print(data.shape)
