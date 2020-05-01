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
#https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
def load_data(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = scale(pd.read_pickle(train_X_fp))
    train_y = pd.read_pickle(train_y_fp)
    oversample = SMOTE()
    print('fitting smote')
    train_X, train_y = oversample.fit_resample(train_X, train_y.ravel())
    print('done smote')
    train_y_bin = to_categorical(train_y)
    train = pd.merge(train_X, train_y)
    train_majority = train[train.fully_funded==0]
    train_minority = train[train.fully_funded==1]
    print(train_majority.shape)
    print(train_majority.shape)
    test = pd.read_pickle(test_fp) 
    with open('train_X_smote.pickle', 'wb') as f:
        pickle.dump(train_X, f, protocol=4)
    print("wrote train X")
    with open('train_y_smote.pickle',  'wb') as f:
        pickle.dump(train_y, f)
    train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)
    return train_X_new, valid_X_new, train_y_new, valid_y_new

load_data()