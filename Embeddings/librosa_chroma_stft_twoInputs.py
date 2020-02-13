import librosa
import numpy as np

#python wave library embeddings
def getEmbedding(filenames):
    #the output is a list of files
    output = []

    #load audio time series of each sample into loaded
    #loads the sample rate of each sample into sr
    loaded = []
    sr = []
    for path in filenames:
        x , y = librosa.load(path)
        loaded.append(x)
        sr.append(y)

    maxLen = 0
    for x in loaded:
        if (len(x) > maxLen):
            maxLen = len(x)

    #extracts features from each audio time series and loads into extracted
    #also finds the largest shape of extracted data
    maxLenAX = 0
    maxLenAY = 0
    maxLenBX = 0
    maxLenBY = 0
    
    a = librosa.feature.chroma_stft(loaded[0], sr[0])
    b = librosa.feature.chroma_stft(loaded[1], sr[1])

    aShapeX, aShapeY = a.shape
    bShapeX, bShapeY = b.shape
    if (aShapeX > maxLenAX):
        maxLenAX = aShapeX
    if (bShapeX > maxLenBX):
        maxLenBX = bShapeX
    if (aShapeY > maxLenAY):
        maxLenAY = aShapeY
    if (bShapeY > maxLenBY):
        maxLenBY = bShapeY

    #pads extracted data
    padded = []
    a = np.pad(a, ((0,0), (maxLenAX-aShapeX,maxLenAY-aShapeY)), mode='constant')
    padded.append(a)
    b = np.pad(b, ((0,0), (maxLenBX-bShapeX,maxLenBY-bShapeY)), mode='constant')
    padded.append(b)

    label = loaded
    data = np.dstack((padded))
    data = np.rollaxis(data,-1)
    data = np.expand_dims(data, axis=0)

    #average
    c = (a+b)/2

    return c

import os
#read in files
path = "../Data/"

filenames = []

filenames.append(path+"bd_909dwsd.wav")
filenames.append(path+"bd_chicago.wav")

getEmbedding(filenames)