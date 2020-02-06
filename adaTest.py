import os
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

