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
from sklearn.model_selection import RandomizedSearchCV

#No. of trees
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

#No. of features for split
max_features = ['auto', 'sqrt']

#max no. of levels
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

#Min no. of samples for split
min_samples_split = [2, 5, 10, 15, 100]

#Min no. of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

#random grid
random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }


#Random forest regressor with random grid
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, scoring='neg_mean_squared_error', verbose=2, random_state=7, cv=5)

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
rf_random.fit(X_train, y_train)
timer(start_time)


print(rf_random.best_score_, rf_random.best_params_)
predictions = rf_random.predict(X_test)
sns.histplot(y_test-predictions)

#Accuracy metrics
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#pickle
import pickle
#opening file to store data
file = open('RandomForest_regression.pkl', 'wb')

#dumping info to file 
pickle.dump(rf_random, file)



