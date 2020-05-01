# References used:
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
# Parameter tuning: https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

import numpy as np 
import sklearn
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pandas as pd 
import random


np.random.seed(819)

def load_data(train_X_fp = "/Users/madhuri/Desktop/10701/project/mylocalcode/train_X_smoteESSAY.pickle", train_y_fp = "/Users/madhuri/Desktop/10701/project/mylocalcode/train_y_smoteESSAY.pickle"):
    train_X = pd.read_pickle(train_X_fp)    #smote with has_essays
    train_y = pd.read_pickle(train_y_fp) #smote with has_essays
    # test = pd.read_pickle(test_fp) 
    train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)
    return train_X_new, valid_X_new, train_y_new, valid_y_new

def tune_n_estimators(train_X, train_y, valid_X, valid_y):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 300, 400, 500, 600, 750]
    train_results = []
    test_results = []
    for estimator in n_estimators:
        print("Num estimators: %d" % estimator)
        rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
        rf.fit(train_X, train_y.values.ravel())
        train_pred = rf.predict(train_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        print("train results roc_auc values", train_results)
        y_pred = rf.predict(valid_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(valid_y, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
        print("test results roc_auc values", test_results)
    line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
    line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()
    maxind = test_results.index(max(test_results)) 
    return n_estimators[maxind]

def tune_max_depth(train_X, train_y, valid_X, valid_y, n_estims):
    max_depths = np.linspace(1, n_estims, n_estims, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        print("Depth: %d" % max_depth)
        rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
        rf.fit(train_X, train_y)
        train_pred = rf.predict(train_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        print("train results roc_auc values", train_results)
        y_pred = rf.predict(valid_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(valid_y, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
        print("test results roc_auc values", test_results)
    line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
    line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()
    maxind = test_results.index(max(test_results)) 
    return max_depths[maxind]


def tune_max_features(train_X, train_y, valid_X, valid_y):
    max_features = list(range(1,train_X.shape[1]))
    train_results = []
    test_results = []
    for max_feature in max_features:
        print("Num max features: %d" % max_feature)
        rf = RandomForestClassifier(max_features=max_feature)
        rf.fit(train_X, train_y)
        train_pred = rf.predict(train_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(valid_X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(valid_y, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
    line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('max features')
    plt.show()
    maxind = test_results.index(max(test_results)) 
    return max_features[maxind]

def rf_classifier(train_X, train_y, valid_X, valid_y):
    nests = tune_n_estimators(train_X, train_y, valid_X, valid_y) #500
    md = tune_max_depth(train_X, train_y, valid_X, valid_y, nests) 
    mf = tune_max_features(train_X, train_y, valid_X, valid_y)
    print("final n_estimators optimal", nests)
    print("-------------------------------------------------------------------------------")
    model = RandomForestClassifier(n_estimators=nests, bootstrap = True, random_state = 819) # using default max features, max depth
    print(type(train_y))
    model.fit(train_X, np.ravel(train_y))

    y_pred_train = model.predict(train_X)
    y_pred_valids = model.predict(valid_X)
 
    cm = confusion_matrix(train_y, y_pred_train)
    print("Random forest confusion matrix on training set:\n", cm)
    cm1 = confusion_matrix(valid_y, y_pred_valids)
    print("Random forest confusion matrix on validation set:\n", cm1)
    # print(roc_auc_score(valid_y, y_pred_probs))

    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # print(roc_auc)
    # evaluate_model(y_preds, y_pred_probs, )
    train_X = pd.DataFrame(train_X)
    features = list(train_X.columns)
    fi_model = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending = False)
    fi_model.head(10)

    
train_X_new, valid_X_new, train_y_new, valid_y_new = load_data()
rf_classifier(train_X_new, train_y_new, valid_X_new, valid_y_new)