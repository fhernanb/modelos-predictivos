# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:31:37 2020

@author: fhernanb
"""

"""
En este script se muestra como calcular el Cohen's kappa coefficient
"""

# Librerías a usar
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

# Creando los datos
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

# Calculando accuracy y kappa
accuracy_score(y_true, y_pred)
cohen_kappa_score(y_true, y_pred)

