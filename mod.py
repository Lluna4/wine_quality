import sklearn
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pickle", "rb"))
a = "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"
a = a.split(";")
a = np.array(a).reshape(1, -1)
print(model.predict(a))


