#############
#Citations:
# https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
############
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
from sklearn import preprocessing
import pickle


np.random.seed(819)

def load_data(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = pd.read_pickle(train_X_fp)
    columns = pd.read_pickle("train_X.pickle").columns
    ss = preprocessing.StandardScaler()
    train_X = pd.DataFrame(ss.fit_transform(train_X), columns=columns)
    train_y = pd.read_pickle(train_y_fp)
    # train_y_bin = to_categorical(train_y)
    test = pd.read_pickle(test_fp)
    test = pd.DataFrame(ss.fit_transform(test), columns=columns)
    train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)

    # print(np.shape(train_X_new))
    # print(np.shape(train_X_new))
    return train_X_new, valid_X_new, train_y_new, valid_y_new, test

# def load_data(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
#     train_X = scale(pd.read_pickle(train_X_fp))
#     train_y = pd.read_pickle(train_y_fp)
#     oversample = SMOTE()
#     print('fitting smote')
#     train_X, train_y = oversample.fit_resample(train_X, train_y)
#     print('done smote')
#     train_y_bin = to_categorical(train_y)
#     train = pd.concat([pd.DataFrame(train_X), pd.DataFrame(train_y)], axis = 1)
#     # train = pd.merge(train_X, train_y, axis = 1)
#     train_majority = train[train.fully_funded==0]
#     train_minority = train[train.fully_funded==1]
#     print(train_majority.shape)
#     print(train_majority.shape)
#     test = pd.read_pickle(test_fp)
#     with open('train_X_smoteESSAY.pickle', 'wb') as f:
#         pickle.dump(train_X, f, protocol=4)
#     print("wrote train X")
#     with open('train_y_smoteESSAY.pickle',  'wb') as f:
#         pickle.dump(train_y, f)
#     print("wrote train y")
#     train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)
#     return train_X_new, valid_X_new, train_y_new, valid_y_new
#

def simple_ff_nn():
    model = Sequential()
    model.add(Dense(20, input_dim = 118, activation = 'relu', use_bias=True))
    #model.add(Dense(10, activation = 'relu', use_bias=True))
    model.add(Dense(5, activation = 'relu', use_bias=True))
    model.add(Dense(1, activation = 'sigmoid', use_bias = True))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# train_X_new, valid_X_new, train_y_new, valid_y_new, test = load_data()
# my_nn = simple_ff_nn()

# history = my_nn.fit(train_X_new, train_y_new, validation_data = (valid_X_new, valid_y_new), epochs = 1000, batch_size = 100000)
test_fp = "test_set.pickle"
train_fp = "train_X.pickle"
ss = preprocessing.StandardScaler()
test = pd.read_pickle(test_fp)
train = pd.read_pickle(train_fp)
test_fit = test.drop(['projectid','title', 'fully_funded'], axis = 1)
columns = test_fit.columns
print(train.columns)

print(test_fit[['has_essay']])
fp = "nn_model_BCE.pickle"
with open(fp, 'rb') as file:
    my_nn = pickle.load(file)
test_fit = pd.DataFrame(ss.fit_transform(test_fit), columns=columns)
test_preds = my_nn.predict(test_fit)
np.savetxt("final_test_preds.csv", test_preds, delimiter=",")
test.to_csv("test.csv")

print(test_preds)


# y_pred_valid = my_nn.predict(valid_X_new)
# y_pred_train = my_nn.predict(train_X_new)
#
# # print(valid_y_new)
# # print(train_y_new)
# np.savetxt("nnpred_valid.csv", y_pred_valid, delimiter=",")
# np.savetxt("nnpred_train.csv", y_pred_train , delimiter=",")
#
# np.savetxt("nn_y_valid.csv", valid_y_new , delimiter=",")
# np.savetxt("nn_y_trainY.csv", train_y_new, delimiter=",")
#
# # # valid_X_new['y'] = valid_y_new
# # # valid_X_new['y_hat'] = y_pred_valid
# valid_X_new.to_csv("full_valid_nn.csv")
# print("wrote valid")
# # # train_X_new['y'] = train_y_new
# # # train_X_new['y_hat'] = y_pred_train
# # train_X_new.to_csv("full_train_nn.csv")
# # # a = accuracy_score(y_pred,valid_y_new)
# # # print('Accuracy is:', a*100)



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("nnloss.png")
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("nn.png")
plt.show()


