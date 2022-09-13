import pandas as pd
import numpy as np

base = pd.read_csv('ECO050TT.csv', encoding = "ISO-8859-1")

#REGRESSAO PASSAGENS X VALOR
X = base.iloc[:, 0:1].values
y = base.iloc[:,[2]].values

import numpy as np
X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# b0
intercept = regressor.intercept_

# b1
coef = regressor.coef_

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title ("Regressão linear simples")
plt.xlabel("TotalPassagem")
plt.ylabel("Valor")

total_passagem = 600000
previsao1 = regressor.intercept_ + regressor.coef_ * total_passagem 
previsao2 = regressor.predict(np.array(total_passagem ).reshape(1, -1))

score = regressor.score(X,y)

from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()
