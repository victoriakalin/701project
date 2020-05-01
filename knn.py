###########
# Citation: https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
###########
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle



def load_data(train_X_fp = "train_X.pickle", test_fp = "test_set.csv.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = pd.read_pickle(train_X_fp)
    ss = preprocessing.StandardScaler()
    train_X = pd.DataFrame(ss.fit_transform(train_X), columns=train_X.columns)
    train_y = pd.read_pickle(train_y_fp)
    # train_y_bin = to_categorical(train_y)
    test = pd.read_pickle(test_fp)
    train_X_new, valid_X_new, train_y_new, valid_y_new = train_test_split(train_X, train_y, test_size = 0.20, random_state = 701)
    # print(np.shape(train_X_new))
    # print(np.shape(train_X_new))
    return train_X_new, valid_X_new, train_y_new, valid_y_new

def knn_cv(K = 10, train_X_fp = "train_X.pickle", train_y_fp = "train_y_set.pickle"):
    train_X = pd.read_pickle(train_X_fp)
    ss = preprocessing.StandardScaler()
    train_X = pd.DataFrame(ss.fit_transform(train_X), columns=train_X.columns)
    train_y = pd.read_pickle(train_y_fp)
    print("Read in files!")
    k_range = range(1, K)

    k_scores = []

    # loop through  k
    for k in k_range:
        print("start: ", k)
        # Make model
        knn = KNeighborsClassifier(n_neighbors=k, algorithm= "kd_tree", p = 2)
        # get cross_val_score for KNeighborsClassifier with k neighbours
        print("fit")
        scores = cross_val_score(knn, train_X, np.ravel(train_y), cv=5, scoring='accuracy')
        print("done cv")
        # append mean of scores for k neighbors to k_scores list
        k_scores.append(scores.mean())
        print("done: ", k)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')

def knn_tune(train_X_new, valid_X_new, train_y_new, valid_y_new, k = 50, K = 1000, step = 25):
    k_range = range(k, K, step)
    k_scores = []
    # loop through  k
    for k in k_range:
        print("start: ", k)
        # Make model
        knn = KNeighborsClassifier(n_neighbors=k, algorithm= "kd_tree", p = 2)
        # get cross_val_score for KNeighborsClassifier with k neighbours
        print("fit")
        knn.fit(train_X_new, np.ravel(train_y_new))
        print("done fit")
        train_accur = knn.score(train_X_new, train_y_new)
        valid_accur = knn.score(valid_X_new, valid_y_new)
        print("Train Accuracy: ", train_accur)
        print("Validation Accuracy: ", valid_accur)
        k_scores.append(valid_accur)
        print("done: ", k)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Validation Accuracy')


# knn_cv(10)
#
# #
train_X_new, valid_X_new, train_y_new, valid_y_new = load_data()
print("data loaded")
# # knn_tune(train_X_new, valid_X_new, train_y_new, valid_y_new)
knn = KNeighborsClassifier(n_neighbors=50, algorithm= "kd_tree", p = 2)
# print("made!")
# # print(train_X_new.head(100))
# knn.fit(train_X_new, np.ravel(train_y_new))
# print("fit!")
# fp = "knn_model.pickle"
# with open(fp, 'wb') as file:
#     pickle.dump(knn, file)
# print("dumped!")
with open('knn_model.pickle', 'rb') as f:
    knn = pickle.load(f)
print("model loaded!")
print("Validation Accuracy: ", knn.score(valid_X_new, valid_y_new))
print("Train Accuracy: ", knn.score(train_X_new, train_y_new))

