import pandas as pd
import numpy as np

base = pd.read_csv('ECO050TT4.csv', encoding = "ISO-8859-1")

X = base.iloc[:, 0:1].values
y = base.iloc[:,1].values
total_passagem = 600000

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X, y)
score = regressor.score(X, y)

import numpy as np
X_teste = np.arange(min(X), max(X), 50)
X_teste = X_teste.reshape(-1,1)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')
plt.title('Regressão com random forest')
plt.xlabel('TotalPassagem')
plt.ylabel('Evasao')

random_forest = regressor.predict(np.array(total_passagem).reshape(1, -1))