# Exploring Lasso Estimation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

np.random.seed(1234)
n = 10000

eps = np.random.normal(0, 1, n)
x1 = np.random.uniform(-1, 1, n)
x2 = np.random.uniform(-1, 1, n)
x3 = np.random.uniform(-1, 1, n)
x4 = np.random.uniform(-1, 1, n)
x5 = np.random.uniform(-1, 1, n)

y = x1 + 2 * x4 + x3 + \
    x1 * x2 + x1 * x4 + x2 * x3 + x2 * x4 + x3 * x4 + x3 * x5 + eps

dat = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

n_train = 20

train_rows = np.random.choice(n, n_train, replace=False)

train = dat.iloc[train_rows]
test = dat.drop(train_rows)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train = poly.fit_transform(train[['x1', 'x2', 'x3', 'x4', 'x5']])
X_test = poly.transform(test[['x1', 'x2', 'x3', 'x4', 'x5']])

lasso1 = Lasso(alpha=1)
lasso1.fit(X_train, train['y'])
lasso1_coef = lasso1.coef_

print("Lasso Model with lambda = 1")
print("Intercept:", lasso1.intercept_)
print("Coefficients:", lasso1_coef)

lasso0_1 = Lasso(alpha=0.1)
lasso0_1.fit(X_train, train['y'])
lasso0_1_coef = lasso0_1.coef_

print("\nLasso Model with lambda = 0.1")
print("Intercept:", lasso0_1.intercept_)
print("Coefficients:", lasso0_1_coef)

lambdas = [0.001, 0.01, 0.1, 0.5]

model_li = {}
for lam in lambdas:
    model = Lasso(alpha=lam)
    model.fit(X_train, train['y'])
    model_li[lam] = model

yhat_li = {}
for lam, model in model_li.items():
    yhat_li[lam] = model.predict(X_test)

rmse_li = {}
for lam, yhat in yhat_li.items():
    rmse_li[lam] = np.sqrt(mean_squared_error(test['y'], yhat))

print("\nRoot Mean Squared Error for different lambdas:")
print(rmse_li)
