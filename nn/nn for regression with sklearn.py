# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:29:45 2020

@author: fhernanb
"""

"""
En este ejemplo se muestra como usar nn para regresion.

El ejemplo esta basado en http://uc-r.github.io/ann_regression
Los datos del ejemplo están disponibles en un repositorio de github.
"""

# Librerías a usar
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# Leyendo los datos
file = "https://raw.githubusercontent.com/fhernanb/datos/master/datos_regresion_mult_redes.txt"
datos = pd.read_csv(file, comment='#', delimiter='\t')
datos.head()

# Creando X e y
y = datos["y"]
X = datos[["x1", "x2"]]

# Para escalar los valores de X
scaledX = scale.fit_transform(X)

# Creando train y test
X_train, X_test, y_train, y_test = train_test_split(scaledX, y, 
                                                    test_size=0.20, 
                                                    random_state=42)

# Creando el modelo de interés ------------------------------------------------

# Para detalles sobre la función que vamos a usar visite:
import webbrowser
webbrowser.open('https://tinyurl.com/y9efl97l')

# Para definir el modelo
mod = MLPRegressor(solver='adam', 
                   max_iter=1500,
                   alpha=1e-5,
                   activation='logistic',
                   hidden_layer_sizes=(14, 10, 5, 4), 
                   learning_rate='adaptive',
                   random_state=1)

# Para entrenar el modelo
mod.fit(X_train, y_train)

# Estimando y usando los datos de test
y_hat = mod.predict(X_test)

# Para ver algunas medidas de desempeño
metrics.r2_score(y_test, y_hat)
metrics.mean_squared_error(y_test, y_hat)

