from __future__ import print_function

import keras
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributed, BatchNormalization

from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.models import load_model

# Embedding
from DATA.Tailor_Test_Data10 import tailor_test_batch
from DATA.Tailor_Train_Data10 import tailor_train_batch

maxlen = 5000
# Convolution
kernel_size = 3
filters = 18
pool_size = 2
# LSTM
lstm_output_size = 128
# Training
batch_size = 60
epochs = 6

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
def CNN_LSTM_train():
	print('Loading data...')
	x_train, y_train = tailor_train_batch()
	print(len(x_train), 'train sequences')

	print('Pad sequences (samples x time)')
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	print('x_train shape:', x_train.shape)

	print('Build model...')
	model = Sequential()
	model.add(Conv1D(filters,
	                 kernel_size,
	                 padding='same',
	                 strides=1))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size=pool_size))
	model.add(Conv1D(36,
	                 kernel_size,
	                 padding='same',
	                 strides=1))
	model.add(MaxPooling1D(pool_size=pool_size))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(LSTM(lstm_output_size))
	model.add(Dense(7))
	model.add(Activation('softmax'))
	adam = Adam(0.003)
	model.compile(loss='sparse_categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])

	print('Train...')
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs)
	model.save('CNN_LSTM_model_v11.h5')   # HDF5 file

def test():
	keras.backend.clear_session()
	model = load_model('CNN_LSTM_model_v9.h5')
	x_test, y_test = tailor_test_batch()
	score, acc = model.evaluate(x_test, y_test, batch_size=461)
	print('Test score:', score)
	print('Test accuracy:', acc)


CNN_LSTM_train()
# test()
