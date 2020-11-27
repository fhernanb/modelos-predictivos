# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:37:49 2020

@author: fhernanb
"""

import numpy as np
from sklearn.model_selection import KFold

# Los datos ficticios
X = np.array([[1.3, 2.5], [3.2, 4.9], [6.1, 7.4], [8.8, 9.3]])
y = np.array([1, 2, 3, 4])

print(X)
print(y)

# Para crear los folds
kf = KFold(n_splits=2)
kf.get_n_splits(X)

# Para ver las particiones
for train_index, test_index in kf.split(X):
    print("Indices TRAIN:", train_index, "Indices TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Matriz X train \n", X_train)
    print("Matriz X de test \n", X_test)
    print("\n")

