import pandas as pd
import numpy as np

base = pd.read_csv('ECO050TT4.csv', encoding = "ISO-8859-1")

X = base.iloc[:, 0:1].values
y = base.iloc[:,1:2].values
total_passagem = 600000

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor()
regressor.fit(X, y)

regressor.score(X, y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('TotalPassagem')
plt.ylabel('Evasao')

previsao = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(np.array(total_passagem).reshape(1, -1))))