# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:36:22 2020

@author: fhernanb
"""

"""
En este ejemplo se desea se usa adaboost para explicar
la variable respuesta y en función de las covariables x1 a x11. los datos 
provienen del ejercicio 9.5 del libro de Montgomery, Peck and Vining (2003).
El paquete MPV (Braun 2019) contiene todos los datos que acompañan al libro.
"""

# Librerías a usar
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/table_b3_MPV.txt"
datos = pd.read_csv(file, comment='#', delimiter='\t')
print(datos.head())

# Exploremos las filas 22 y 24 porque hay nan
datos.iloc[[22, 24], ]

# Vamos a eliminar las filas que tienen nan
datos = datos.dropna()

# Construyendo X e y
X = datos[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']]
y = datos["y"]

# Para entrenar el modelo
mod = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=2),
                        n_estimators=180, 
                        learning_rate=1, 
                        loss='square', 
                        random_state=0)
mod.fit(X, y)

# Estimando y usando los datos de entrenamiento
y_hat = mod.predict(X)

# Para ver algunas medidas de desempeño
metrics.r2_score(y_true=y, y_pred=y_hat)
metrics.mean_squared_error(y_true=y, y_pred=y_hat)

# Agregando el modelo estimado al diagrama originar
plt.scatter(x=y, y=y_hat, color='black', alpha=0.55)
plt.xlabel('Y')
plt.ylabel('$\hat{Y}$')
plt.title('Relación entre Y y $\hat{Y}$')
plt.show()


