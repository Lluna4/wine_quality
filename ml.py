import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


a = pd.read_csv('winequality-white.csv', sep=';')
a = pd.DataFrame(a)
a.drop(["fixed acidity", "density", "chlorides", "free sulfur dioxide", "residual sugar", "total sulfur dioxide", "pH",], axis=1)
x = np.array(a.drop(['quality'], axis=1))
y = np.array(a['quality'])
prec = []
best = 0

for i in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    clf = MLPClassifier(hidden_layer_sizes=(25, 12, 6), max_iter=1000).fit(x_train, y_train)
    
    prec.append(clf.score(x_test, y_test))
    if clf.score(x_test, y_test) > best:
        best = clf.score(x_test, y_test)
        with open('model.pkl', 'wb') as f:
            pickle.dump(clf, f)

x = np.arange(10)

plt.plot(x, prec)
plt.show()
"""
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


linear = MLPClassifier(hidden_layer_sizes=(25, 12, 6), max_iter=1000).fit(X_train, y_train)

precision = linear.score(X_test, y_test)
print(precision)"""




