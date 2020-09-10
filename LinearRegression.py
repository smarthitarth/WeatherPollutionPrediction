# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:26:47 2020

@author: hitarth
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Real-Data/Real_Combine.csv')

sns.heatmap(df.isnull(),yticklabels=False, cbar=True, cmap='viridis')

df = df.dropna()

X = df.iloc[:,:-1]
y = df.iloc[:, -1]


#pair plot
sns.pairplot(df)

#heatmap
co = df.corr()
top_co = co.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_co].corr(),annot=True,cmap="RdYlGn")

#featureImportance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)
f_importances = pd.Series(model.feature_importances_, index=X.columns)
f_importances.nlargest(6).plot(kind='barh')
plt.show()

sns.distplot(y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Coeff of determination R^2 is(Train): {}".format(regressor.score(X_train, y_train)))
print("Coeff of determination R^2 is(Test): {}".format(regressor.score(X_test, y_test)))


#cross val score
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)
score.mean()

coeff_df = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])
#Predict
prediction = regressor.predict(X_test)

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
file = open('linear_regression.pkl', 'wb')

#dumping info to file
pickle.dump(regressor, file)