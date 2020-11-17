# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 07:28:47 2020

@author: fhernanb
"""

# Librerías a usar
import pandas as pd

from sklearn.model_selection import train_test_split

# Leyendo los datos
file = 'https://tinyurl.com/k55nnlu'
datos = pd.read_csv(file, sep="\t", skiprows=9)
datos.head()

# Creando X e y
y = datos["peso"]
X = datos[["edad", "altura", "muneca"]]

# Creando train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=42)

# Para explorar train y test
print("\n Esta es la matriz X_train \n", X_train)
print("\n Este es el vector y_train \n", y_train)
print("\n Esta es la matriz X_test \n", X_test)
print("\n Este es el vector y_test \n", y_test)
