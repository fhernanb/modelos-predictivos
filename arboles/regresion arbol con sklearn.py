# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:22:27 2020

@author: fhernanb
"""

"""
En este ejemplo se desea crea un árbol de regresion que explique 
la variable respuesta y en función de las covariables x1 a x11. los datos 
provienen del ejercicio 9.5 del libro de Montgomery, Peck and Vining (2003).
El paquete MPV (Braun 2019) contiene todos los datos que acompañan al libro.
"""

# Librerías a usar
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
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

# Creando el modelo de interés ------------------------------------------------

# Para detalles sobre la función que vamos a usar visite:
import webbrowser
webbrowser.open('https://tinyurl.com/y6rwb6y5')

# Vamos a crear dos modelos alternativos:

# Para definir los modelos
mod1 = DecisionTreeRegressor(criterion='mse', max_depth=2, random_state=0)
mod2 = DecisionTreeRegressor(criterion='mse', max_depth=5, random_state=0)

# Para entrenar los modelos
mod1.fit(X, y)
mod2.fit(X, y)

# Estimando y usando los datos de entrenamiento
y_hat1 = mod1.predict(X)
y_hat2 = mod2.predict(X)

# Para ver algunas medidas de desempeño
metrics.r2_score(y_true=y, y_pred=y_hat1)
metrics.r2_score(y_true=y, y_pred=y_hat2)

metrics.mean_squared_error(y_true=y, y_pred=y_hat1)
metrics.mean_squared_error(y_true=y, y_pred=y_hat2)

# Para dibujar los dos árboles
tree.plot_tree(mod1, filled=True)
tree.plot_tree(mod2, filled=True) 
