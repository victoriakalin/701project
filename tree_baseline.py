
import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


np.random.seed(819)


def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


def run_treeClassifier(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = pd.read_pickle(train_X_fp)
    train_y = pd.read_pickle(train_y_fp)
    test = pd.read_pickle(test_fp) 
    train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20)

    # Model based on Train Data 

    treeCl = DecisionTreeClassifier(random_state=819).fit(train_X_new, train_y_new)
     

    # Predict on Train set and Validation set

    ypredstrain = treeCl.predict(train_X_new) 
    ypredsvalid = treeCl.predict(valid_X_new) 
    treeTrainScore = treeCl.score(train_X_new, train_y_new) 
    treeValidScore = treeCl.score(valid_X_new, valid_y_new) 
    print("Full Depth Tree Mean train error:", treeTrainScore)
    print("Full Depth Tree Mean validation error:", treeValidScore)
    print("Full Depth Tree Confusion matrix on Training Set\n", confusion_matrix(train_y_new, ypredstrain))
    print("Full Depth Tree Confusion matrix on Validation Set\n", confusion_matrix(valid_y_new, ypredsvalid))



    # Depth of our fully grown tree? Should be 62 with seed(819)

    full_depth = treeCl.get_depth()
    print("Fully grown depth: ", full_depth)


    # Feature importance of the full_depth tree

    featureimportances = dict(zip(train_X_new.columns, treeCl.feature_importances_))
    fimpstring = "Full Depth Tree Feature Importances: \n"
    orderedfeatureimportances = {k: v for k, v in sorted(featureimportances.items(), key=lambda item: item[1], reverse = True)}
    for pair in orderedfeatureimportances.items():
        fimpstring = fimpstring + "%s: %f\n" % (pair[0], pair[1])    
    writeFile("fimp.txt", fimpstring)


    # Tuning for optimal tree max depth we should use

    trying_max_depths = list(range(full_depth))
    trainaccuracies = list()
    validaccuracies = list()
    for d in (trying_max_depths[1:]):
        treeCl_d = DecisionTreeClassifier(max_depth=d, random_state=819).fit(train_X_new, train_y_new)
        trainaccuracies.append(treeCl_d.score(train_X_new, train_y_new))
        validaccuracies.append(treeCl_d.score(valid_X_new, valid_y_new))
    errors = [1 - x for x in validaccuracies]
    optimal_depth_index = errors.index(min(errors))
    optimal_depth = trying_max_depths[optimal_depth_index]
    print("Optimal max depth: ", optimal_depth)


    # New Model for Tree with Optimal Max depth:

    optTreeCl = DecisionTreeClassifier(random_state=819, max_depth = optimal_depth).fit(train_X_new, train_y_new)
     

    # Predict on Train set and Validation set

    optypredstrain = optTreeCl.predict(train_X_new) 
    optypredsvalid = optTreeCl.predict(valid_X_new)
    opttreeTrainScore = optTreeCl.score(train_X_new, train_y_new) 
    opttreeValidScore = optTreeCl.score(valid_X_new, valid_y_new) 
    print("Optimal depth tree Mean train error:", opttreeTrainScore)
    print("Optimal depth tree Mean validation error:", opttreeValidScore)
    print("Optimal depth tree Confusion matrix on Training Set\n", confusion_matrix(train_y_new, optypredstrain))
    print("Optimal depth tree Confusion matrix on Validation Set\n", confusion_matrix(valid_y_new, optypredsvalid))


    # Feature importance on Optimal Depth tree: 

    optfeatureimportances = dict(zip(train_X_new.columns, treeCl.feature_importances_))
    optfimpstring = "Optimal Depth Tree Feature Importances: \n"
    optorderedfeatureimportances = {k: v for k, v in sorted(optfeatureimportances.items(), key=lambda item: item[1], reverse = True)}
    for pair in optorderedfeatureimportances.items():
        optfimpstring = optfimpstring + "%s: %f\n" % (pair[0], pair[1])    
    writeFile("optfimp.txt", optfimpstring)


    # Plot of Training and Validation Errors during Max_Depth Parameter Tuning

    plt.plot(trying_max_depths[1:], [1-x for x in trainaccuracies], label = "Training Error")
    plt.title('Tuning Decision Tree Max Depth')
    plt.xlabel('Max_depth of Decison Tree Classifier')
    plt.ylabel('ERROR RATE')
    plt.plot(trying_max_depths[1:], [1-x for x in validaccuracies], label = "Validation Error")
    plt.xlabel('Max_depth of Decison Tree Classifier')
    plt.legend()
    plt.show()


    
run_treeClassifier()


