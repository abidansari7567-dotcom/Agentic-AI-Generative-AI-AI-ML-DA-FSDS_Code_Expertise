import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\data science\class\Salary_Data.csv")
print("Dataset shape:", dataset.shape)

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=0)


x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Exprience (Training set)')
plt.xlabel('Years of Exprience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Saalry vs Exprience (test set)')
plt.xlabel('Year of Exprience')
plt.ylabel('Salary')
plt.show()

print(f"intercept: {regressor.intercept_}")
print(f"coefficient: {regressor.coef_}")

bias = regressor.score(x_train, y_train)
print(bias)


variance = regressor.score(x_test,y_test)
print(variance)


comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)



dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].mode()

dataset.describe()

dataset.var()

dataset.std()

dataset.corr()

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1- SSR/SST
print(r_square)

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)
