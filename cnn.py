import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
import pandas as pd
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import resample
from sklearn.preprocessing import scale
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pickle

#https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
#https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
#https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))
 
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])
 
# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded
 
# define the model
def define_model(length, vocab_size):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	
	return model
 

with open('xtrpad.pickle', 'rb') as handle:
    X_train_new = pickle.load(handle)
with open('xvapad.pickle', 'rb') as handle:
    valid_X_new = pickle.load(handle)


with open('ytrpad.pickle', 'rb') as handle:
    train_y_new = pickle.load(handle)
with open('yvapad.pickle', 'rb') as handle:
    valid_y_new = pickle.load(handle)

with open('vs.pickle', 'rb') as handle:
    vocab_size = pickle.load(handle)
with open('ml.pickle', 'rb') as handle:
    max_length = pickle.load(handle)



# define model
my_nn = define_model(max_length, vocab_size)
# fit model
#history = my_nn.fit([X_train_new,X_train_new,X_train_new], train_y_new, epochs=10, batch_size=1000)
# save the model
my_nn.load_weights('my_model_weights.h5')
print("VALIDATION SET PREDICTIONS:")
#print(history.history['accuracy'])
#print(history.history['val_accuracy'])


y_pred = my_nn.predict([valid_X_new,valid_X_new,valid_X_new])
print(np.mean((valid_y_new == np.where(y_pred>0.5,1,0))))
#print(history.history['accuracy'])
#print(history.history['val_accuracy'])
with open('valpreds.pickle', 'wb') as handle:
    pickle.dump(y_pred, handle)

print(y_pred )
con_mat = confusion_matrix(valid_y_new, np.where(y_pred>0.5,1,0))
print(con_mat)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()