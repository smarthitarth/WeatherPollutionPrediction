# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 02:29:13 2020

@author: hitarth
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Real-Data/Real_Combine.csv')

df = df.dropna()

X = df.iloc[:,:-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7)

#Decision tree
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(criterion="mse")
dtree.fit(X_train, y_train)

print("Coeff R^2(train): {}".format(dtree.score(X_train, y_train)))
print("Coeff R^2(test): {}".format(dtree.score(X_test, y_test)))

#Score
score = cross_val_score(dtree, X,y,cv=5)
score.mean()

#visualization
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

features = list(df.columns[:-1])
print(features)

import os 
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())

prediction = dtree.predict(X_test)
plt.scatter(y_test, prediction)



