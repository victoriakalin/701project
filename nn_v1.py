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

np.random.seed(819)

def load_data(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = pd.read_pickle(train_X_fp)
    train_y = pd.read_pickle(train_y_fp)
    # train_y_bin = to_categorical(train_y)
    test = pd.read_pickle(test_fp) 
    train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)
    return train_X_new, valid_X_new, train_y_new, valid_y_new

def simple_ff_nn():
    model = Sequential()
    model.add(Dense(20, input_dim = 120, activation = 'relu'))
    model.add(Dense(5, activation = 'relu'))  
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


train_X_new, valid_X_new, train_y_new, valid_y_new = load_data()
my_nn = simple_ff_nn()

history = my_nn.fit(train_X_new, train_y_new, validation_data = (valid_X_new, valid_y_new), epochs = 100, batch_size = 100)

# y_pred = my_nn.predict(valid_X_new)
# a = accuracy_score(y_pred,valid_y_new)
# print('Accuracy is:', a*100)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()