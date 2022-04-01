import sklearn
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
a = "5.5;0.29;0.3;1.1;0.022;20;110;0.98869;3.34;0.38;12.8"
a = a.split(";")
a = np.array(a).reshape(1, -1)
print(model.predict(a))


