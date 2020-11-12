# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:22:27 2020

@author: fhernanb
"""

# -------------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) para mostrar
# el uso de svm en regresion
# -------------------------------------------------------------------------

# Librerías a usar
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor


# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/table_b3_MASS.txt"
datos = pd.read_csv(file, comment='#', delimiter='\t')
datos.head()

# Exploremos las filas 22 y 24 porque hay nan
datos.iloc[[22, 24], ]

# Vamos a eliminar las filas que tienen nan
datos = datos.dropna()

# Construyendo X e y
X = datos[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']]
y = datos["y"]

# Creando el modelo arbol --------------------------------------------------
# Vamos a crear dos modelos alternativos

mod1 = DecisionTreeRegressor(criterion='mse', max_depth=3, random_state=0)
mod2 = DecisionTreeRegressor(criterion='mse', max_depth=5, random_state=0)

mod1.fit(X, y)
mod2.fit(X, y)

# Para ver el score
mod1.score(X, y)
mod2.score(X, y)


# Estimando y usando los datos de entrenamiento
y_hat1 = mod1.predict(X)
y_hat2 = mod2.predict(X)

# Funcion para calcular MSE
def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

mse(y, y_hat1) # valor de mse
mse(y, y_hat2) # valor de mse

# Para dibujar los dos árboles
tree.plot_tree(mod1) 
tree.plot_tree(mod2) 
