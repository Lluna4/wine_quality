import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle


a = pd.read_csv('winequality-white.csv', sep=';')
a = pd.DataFrame(a)
a.drop(["fixed acidity", "density", "chlorides", "free sulfur dioxide", "residual sugar", "total sulfur dioxide", "pH",], axis=1)
x = np.array(a.drop(['quality'], axis=1))
y = np.array(a['quality'])
prec = []
for i in range(100):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    precision = linear.score(X_test, y_test)
    print(precision)
    prec.append(precision)
    if precision >= 0.367:
        with open("model.pickle", "wb") as f:
            pickle.dump(linear, f)

y = np.arange(0, len(prec))

plt.plot(y, prec)
plt.show()

