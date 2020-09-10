# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:25:36 2020

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
from sklearn.ensemble import RandomForestRegressor
rRegressor = RandomForestRegressor()
rRegressor.fit(X_train, y_train)

print("Coeff R^2(train): {}".format(rRegressor.score(X_train, y_train)))
print("Coeff R^2(test): {}".format(rRegressor.score(X_test, y_test)))

#Score
score = cross_val_score(rRegressor, X,y,cv=5)
score.mean()

prediction = rRegressor.predict(X_test)
plt.scatter(y_test, prediction)


#Hyper parameter Tuning

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

#pickle
import pickle
#opening file to store data
file = open('DecisionTree_regression.pkl', 'wb')

#dumping info to file
pickle.dump(random_search, file)



