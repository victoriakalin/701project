import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression

def run_baseline(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = pd.read_pickle(train_X_fp)
    train_y = pd.read_pickle(train_y_fp)
    test = pd.read_pickle(test_fp)

    # print(test)
    print(train_X.columns)
    print(train_y.head())
    logreg = LogisticRegression().fit(train_X, train_y)

    print("done model")
    score = logreg.score(train_X, train_y)
    print(score)
    # print(logreg.predict(test))


run_baseline()