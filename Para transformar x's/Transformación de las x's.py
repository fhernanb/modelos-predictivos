# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:19:53 2020

@author: fhern
"""

"""
En este script se muestran los ejemplos de 
https://scikit-learn.org/stable/modules/preprocessing.html
para procesar los datos
"""


from sklearn import preprocessing
import numpy as np

# Datos ficticios para ilustrar -----------------------------------------------
X = np.array([[2.5, 4.5, 6.7],
              [3.1, 5.6, 7.2],
              [2.8, 4.4, 6.5],
              [2.6, 5.8, 6.9]])

# Caso 1: estandarización (media 0 y varianza 1)-------------------------------

# Estandarizando
X1 = preprocessing.scale(X)

# Para ver los datos transformados
X1

# Verificar el que si los datos transformados tienen media 0 y varianza 1
X1.mean(axis=0)
X1.std(axis=0)

# Caso 2: al intervalo (0, 1) -------------------------------------------------

# Transformando
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X2 = scaler.transform(X)

# Para ver los datos transformados
X2


