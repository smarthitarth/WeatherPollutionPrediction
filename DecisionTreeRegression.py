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


#Hyper parameter optimization
params={
        "splitter":["best", "random"],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4],
        "max_features": ["auto", "log2", "sqrt", None],
        "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70]}

from sklearn.model_selection import GridSearchCV

random_search = GridSearchCV(dtree, param_grid = params, scoring='neg_mean_squared_error', n_jobs=-1, cv=10, verbose=3)


#timer function
def timer(start_time = None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        hour, temp_sec = divmod((datetime.now()-start_time).total_seconds(), 3600)
        mins, sec = divmod(temp_sec, 60)
        print("\n Time taken: %i:%i:%s" % (hour, mins, round(sec, 2)))

from datetime import datetime
start_time = timer(None)
random_search.fit(X, y)
timer(start_time)


print(random_search.best_score_, random_search.best_params_)
predictions = random_search.predict(X_test)
sns.histplot(y_test-predictions)

#Accuracy metrics
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
