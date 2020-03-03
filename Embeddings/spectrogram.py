import librosa #used to load wav to python
import numpy as np #used to pad the arrays
from scipy import signal #used to get the spectrogram

def embed2d(filenames):
	'''
	input is a list of file paths 
	output is a list of spectrograms and a list of related sample rates
	'''

	#load audio time series of each sample into loaded
	#loads the sample rate of each sample into sr
	loaded = []
	sr = []
	for path in filenames:
		x , y = librosa.load(path)
		loaded.append(x)
		sr.append(y)

	#create the spectrograms for each wav file
	#adds the spectorgrams to data
	data = []
	for i in range(0, len(loaded)):
		_, _, Sxx = signal.spectrogram(loaded[i], sr[i])
		data.append(Sxx)

	#finds the maximum data size for padding
	maxLen = 0
	for x in data:
		_, temp = x.shape
		if (temp > maxLen):
			maxLen = temp

	#pads data into padded_data
	padded_data = []
	for x in data:
		x1, temp = x.shape
		arLen = max(x1, temp)
		tempArray = np.zeros((arLen, arLen))
		tempArray[:x1, :x.shape[1]] = x
		padded_data.append(tempArray)

	return padded_data, sr

def embed1d(filenames):
	'''
	input is a list of file paths 
	output is a list of spectrograms and a list of related sample rates
	'''

	#load audio time series of each sample into loaded
	#loads the sample rate of each sample into sr
	loaded = []
	sr = []
	for path in filenames:
		x , y = librosa.load(path)
		loaded.append(x)
		sr.append(y)

	#finds the maximum length
	dmax = 0
	for x in loaded:
		if (len(x) > dmax):
			dmax = len(x)

	# pads the wav file for transer to 1d network 
	padded_data = []
	for x in loaded:
		tempArray = np.zeros(dmax)
		tempArray[:len(x)] = x
		padded_data.append(tempArray)

	return padded_data, sr