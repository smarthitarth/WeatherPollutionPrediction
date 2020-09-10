# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:31:48 2020

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7)

#Linear Regression
lin_regressor = LinearRegression()
mse = cross_val_score(lin_regressor, X, y, scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
print(mean_mse)

#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 5, 10, 20, 30, 35, 40]}
ridge_regressor = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5 )
ridge_regressor.fit(X, y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X_train, y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

prediction = lasso_regressor.predict(X_test)

sns.distplot(y_test-prediction)
plt.scatter(y_test,prediction)


#Accuracy metrics
from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, prediction))
print("MSE: ", metrics.mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, prediction)))



#pickle
import pickle
#opening file to store data
file = open('lasso_regression.pkl', 'wb')

#dumping info to file
pickle.dump(lasso_regressor, file)

