import pandas as pd
import numpy as np

base = pd.read_csv('ECO050TT.csv', encoding = "ISO-8859-1")

X = base.iloc[:, 0:1].values
y = base.iloc[:,[2]].values

# Regressão linear simples
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)
intercept = regressor1.intercept_
# b1
coef = regressor1.coef_

total_passagem = 600000
linear = regressor1.predict(np.array(total_passagem).reshape(1, -1))

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor1.predict(X), color = 'red')
plt.title('Regressão linear')
plt.xlabel('TotalPassagem')
plt.ylabel('Valor')

# Regressão polinomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly, y)
score2 = regressor2.score(X_poly, y)

intercept = regressor2.intercept_
# b1
coef = regressor2.coef_
polinomial = regressor2.predict(poly.transform(np.array(total_passagem).reshape(1, -1)))

plt.scatter(X, y)
plt.plot(X, regressor2.predict(poly.fit_transform(X)), color = 'red')
plt.xlabel('TotalPassagem')
plt.ylabel('Valor')

# b0
