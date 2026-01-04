import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\data science\class\All Dec Class\23rd, 24th, 26th, - Svr, Dtr, Rf, Knn\23rd, 24th, 26th, - Svr, Dtr, Rf, Knn\emp_sal.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.svm import SVR
regressor = SVR(kernel = 'poly', degree= 5)

regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth  or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()