import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#sklearn kfold documentation
def run_baseline(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = scale(pd.read_pickle(train_X_fp))
    train_y = pd.read_pickle(train_y_fp).values
    test = pd.read_pickle(test_fp)
    print(test)
    # print(test)
    #print(train_X.columns)
    #print(train_y.head())
    i = -3
    res = []
    print('start cv')
    #logreg = LogisticRegressionCV().fit(train_X, train_y)
    l = []
    for j in range(6):
        
        score1 = 0
        kf = KFold(n_splits=5, random_state=10701, shuffle=True)
        for train_index, test_index in kf.split(train_X):
            X_train, X_test = train_X[train_index], train_X[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            logreg = LogisticRegression(C = 10**i, max_iter= 500).fit(X_train, y_train)

            print("done model")
            score = logreg.score(X_test, y_test)
            print(score)
            print(i)
            res += [(10**i, score)]
            score1 += score
        l +=[(score1/5, i)]
        i +=1 
            
        print('_______________')
    print(l)
    print(res)
    #print(logreg.predict(test))
    
    

run_baseline()