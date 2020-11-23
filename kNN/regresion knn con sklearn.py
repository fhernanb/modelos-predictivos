# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:21:48 2020

@author: fhern
"""

"""
En este ejemplo vamos a utilizar la base de datos Cars93
del paquete MASS para estimar el precio del auto en funcion del
AirBags, DriveTrain y Origin
"""

# Librerías a usar
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/Cars93_MASS.txt"
datos = pd.read_csv(file, comment='#', delimiter='\t')
datos.head()


# Vamos a seleccionar las variables de interes
datos = datos[["Price",
               "AirBags",
               "DriveTrain",
               "Origin"]]

# Creando las variables dummy para las variables cualitativas
datos_dum = pd.get_dummies(datos, 
                           columns=['AirBags', 'DriveTrain', 'Origin'], 
                           drop_first=True)

# Creando X e y
y = datos_dum["Price"]
X = datos_dum.drop('Price', axis=1)

# Creando train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=42)

# Creando el modelo de interés ------------------------------------------------

# Para definir el modelo
mod = KNeighborsRegressor(n_neighbors=3, algorithm='brute')

# Para entrenar el modelo
mod.fit(X_train, y_train)

# Estimando y usando los datos de test
y_hat = mod.predict(X_test)

# Para ver algunas medidas de desempeño
metrics.r2_score(y_test, y_hat)
metrics.mean_squared_error(y_test, y_hat)

