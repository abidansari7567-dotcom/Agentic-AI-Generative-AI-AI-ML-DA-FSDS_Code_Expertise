import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\data science\class\All Dec Class\23rd, 24th, 26th, - Svr, Dtr, Rf, Knn\23rd, 24th, 26th, - Svr, Dtr, Rf, Knn\emp_sal.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly', degree= 4, gamma= 'auto', C=10)
svr_regressor.fit(x,y)

svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5, weights='distance', leaf_size=30)
knn_reg_model.fit(x, y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)

from sklearn.tree import DecisionTreeRegressor
dtr_reg_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=10, splitter='random')
dtr_reg_model.fit(x,y)

from sklearn.ensemble import RandomForestRegressor
rfr_reg_model = RandomForestRegressor(n_estimators=6, random_state=0)
rfr_reg_model.fit(x,y)

rfr_reg_pred = rfr_reg_model.predict([[6.5]])
print(rfr_reg_pred)
