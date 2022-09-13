import pandas as pd
import numpy as np

base = pd.read_csv('ECO050TT.csv', encoding = "ISO-8859-1")

X = base.iloc[:, 0:1].values
y = base.iloc[:,[2]].values

total_passagem = 600000

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)
score = regressor.score(X, y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com árvores')
plt.xlabel('TotalPassagem')
plt.ylabel('Valor')


import numpy as np
X_teste = np.arange(min(X), max(X), 50)
X_teste = X_teste.reshape(-1,1)
plt.scatter(X, y)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')
plt.title('Regressão com árvores')
plt.xlabel('TotalPassagem')
plt.ylabel('Valor')

arvore_decisao = regressor.predict(np.array(total_passagem).reshape(1, -1))