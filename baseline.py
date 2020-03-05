import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing



def run_baseline(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = pd.read_pickle(train_X_fp).values
    train_y = pd.read_pickle(train_y_fp).values
    test = pd.read_pickle(test_fp).values

    train_X = preprocessing.scale(train_X)
    print("done scale")
    ## Seperate out validate and train set
    train_X, X_valid, train_y, y_valid = train_test_split(train_X, train_y, test_size = 0.2, random_state = 701)
    ##

    # print(test)
    # print(train_X.columns)
    # print(train_y.head())

    ## Run model
    logreg = LogisticRegression(max_iter = 1000).fit(train_X, np.ravel(train_y))
    print("done model")

    ## Predict
    y_train_pred = logreg.predict(train_X)
    y_valid_pred = logreg.predict(X_valid)


    ## Metrics
    train_accur = logreg.score(train_X, train_y)
    print("Train Accuracy:", train_accur)
    test_accur = logreg.score(X_valid, y_valid)
    print("Test Accuracy:", test_accur)


    print("Train CM:", confusion_matrix(train_y, np.ravel(y_train_pred), labels = [0,1]))
    print("Valid CM:", confusion_matrix(y_valid, np.ravel(y_valid_pred), labels = [0,1]))

    with open('y_valid.pickle', 'wb') as f:
        pickle.dump(y_valid, f)
    with open('X_valid.pickle', 'wb') as f:
        pickle.dump(X_valid, f)
    with open('pred_train.pickle', 'wb') as f:
        pickle.dump(y_train_pred, f)
    with open('pred_valid.pickle', 'wb') as f:
        pickle.dump(y_valid_pred, f)


run_baseline()