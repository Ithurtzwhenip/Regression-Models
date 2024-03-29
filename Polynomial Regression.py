import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear Regression

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression

lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression

pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print("The predicted salary is", pred)
