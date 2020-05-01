
#https://towardsdatascience.com/cnn-sentiment-analysis-1d16b7c5a0e7
#https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456

import pandas as pd 
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
essays = pd.read_csv("kdd-cup-2014-predicting-excitement-at-donors-choose/essays.csv")
outcomes = pd.read_csv("kdd-cup-2014-predicting-excitement-at-donors-choose/outcomes.csv")
np.random.seed(819)
total = pd.merge(essays, outcomes, on = "projectid", how = "outer")

train = total[total['fully_funded'].notna()]
train = train.replace({"t":1, "f":0})
test = total[total['fully_funded'].isna()]
#test = test.replace({"t":1, "f":0})
#test = test.dropna()
#test.drop(['fully_funded'], axis = 1)
train = train.dropna()
train_y = train[['fully_funded']]['fully_funded'].values
train_X = train[['essay']]['essay'].values
train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(train_X)
max_length = max(len(s.split()) for s in train_X)
vocab_size = len(tokenizer_obj.word_index) + 1
X_train_tokens = tokenizer_obj.texts_to_sequences(train_X_new)
X_valid_tokens = tokenizer_obj.texts_to_sequences(valid_X_new)

X_train_pad = pad_sequences(X_train_tokens,maxlen = max_length, padding = 'post')
X_valid_pad = pad_sequences(X_valid_tokens,maxlen = max_length, padding = 'post')

print("DONE PREPROCESSING")
import pickle

with open('xtrpad.pickle', 'wb') as handle:
    pickle.dump(X_train_pad, handle)
with open('xvapad.pickle', 'wb') as handle:
    pickle.dump(X_valid_pad, handle)


with open('ytrpad.pickle', 'wb') as handle:
    pickle.dump(train_y_new, handle)
with open('yvapad.pickle', 'wb') as handle:
    pickle.dump(valid_y_new, handle)

with open('vs.pickle', 'wb') as handle:
    pickle.dump(vocab_size, handle)
with open('ml.pickle', 'wb') as handle:
    pickle.dump(max_length, handle)


