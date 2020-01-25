import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, LSTM, Dense, Activation, Input

#returns the LSTM Model
def getModel(length):
	model = Sequential()
	model.add(Input((length,)))
	model.add(LSTM(256,return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(256))
	model.add(Dense(256))
	model.add(Dropout(0.3))
	model.add(Dense(n_vocab))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model