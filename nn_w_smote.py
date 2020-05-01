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
np.random.seed(819)
#https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
#https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
def load_data(train_X_fp = "train_X_smoteESSAY.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_smoteESSAY.pickle"):
    train_X = scale(pd.read_pickle(train_X_fp))
    train_y = pd.read_pickle(train_y_fp)
    #oversample = SMOTE()
    #print('fitting smote')
    #train_X, train_y = oversample.fit_resample(train_X, train_y.ravel())
    #print('done smote')
    #train_y_bin = to_categorical(train_y)
    #train = pd.merge(train_X, train_y)
    #train_majority = train[train.fully_funded==0]
    #train_minority = train[train.fully_funded==1]
    #print(train_majority.shape)
    #print(train_majority.shape)
    #test = pd.read_pickle(test_fp) 
    #with open('train_X_smote.pickle', 'wb') as f:
    #    pickle.dump(train_X, f, protocol=4)
    #print("wrote train X")
    #with open('train_y_smote.pickle',  'wb') as f:
    #    pickle.dump(train_y, f)
    train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)
    return train_X_new, valid_X_new, train_y_new, valid_y_new
import torch
import torch.nn as nn
from torch.autograd import Variable
import keras.backend as K

#https://forums.fast.ai/t/custom-loss-function-that-penalizes-false-positive-for-neural-networks/10678/2
#https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
#https://forums.fast.ai/t/custom-loss-function-that-penalizes-false-positive-for-neural-networks/10678/4
#https://towardsdatascience.com/custom-loss-function-in-tensorflow-2-0-d8fa35405e4e
#
#https://github.com/tensorflow/tensorflow/issues/35403

@tf.function
def weighted_loss(yhats, ys):
    #fpcount = K.mean(yhats  -ys >0)
    fpcount = K.mean(K.map_fn(lambda x: 1.0 if x > 0.5 else 0.0, yhats) - ys > 0)
    rest = K.mean(K.square(yhats - ys))
    return fpcount*2 +  rest

def simple_ff_nn():
    model = Sequential()
    model.add(Dense(50, input_dim = 118, activation = 'relu'))
    model.add(Dense(20, activation = 'relu')) 
    model.add(Dense(5, activation = 'relu'))  
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = weighted_loss, optimizer = 'adam', metrics = ['accuracy'])
    return model


train_X_new, valid_X_new, train_y_new, valid_y_new = load_data()
my_nn = simple_ff_nn()
#history = my_nn.load_model('my_nn_newloss.h5')
history = my_nn.fit(train_X_new, train_y_new,  validation_data = (valid_X_new, valid_y_new), epochs = 300, batch_size = 1000)
# y_pred = my_nn.predict(valid_X_new)
# a = accuracy_score(y_pred,valid_y_new)
# print('Accuracy is:', a*100)
#print(history.history['accuracy'])
#print("VALIDATION SET PREDICTIONS:")
#y_pred = my_nn.predict(valid_X_new)
#print(y_pred )
#my_nn.save('my_nn_newloss.h5')
#with open('valpreds_newloss.pickle', 'wb') as handle:
#    pickle.dump(y_pred, handle)
#con_mat = confusion_matrix(valid_y_new, np.where(y_pred>0.5,1,0))
#print(con_mat)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()