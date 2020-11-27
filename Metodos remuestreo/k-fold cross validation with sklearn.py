# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:00:02 2020

@author: fhernanb
"""

"""
En este ejemplo vamos a utilizar a aplicar k-fold cross validation 
usando la base de datos Auto de ISLR.
Lo vamos a realizar de forma manual y automatica
y vamos a crear un lm para explicar mpg en funcion de 
horsepower y de horsepower^2
Metrica a usar: rmse
"""

# Librerías a usar
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/Auto_ISLR.txt"
datos = pd.read_csv(file, comment='#', delimiter='\t')
datos.head()

# Construyendo X e y
sub_dat = datos["horsepower"].values.reshape(-1, 1) # necesitamos un array de 2D para SkLearn
poly = PolynomialFeatures(degree=2, include_bias=False)
sub_dat = poly.fit_transform(sub_dat)

X = pd.DataFrame(sub_dat, columns=['hp','hp2'])
y = datos["mpg"]

# Definiendo el modelo
mod = linear_model.LinearRegression()

# KFold Cross Validation approach
k = 10
kf = KFold(n_splits=k, shuffle=False)
kf.split(X)  

# Creando los modelos

# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
scores = []

for train_index, test_index in kf.split(X):
    # Split train-test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train the model
    model = mod.fit(X_train, y_train)
    # Append to accuracy_model the accuracy of the model
    y_hat = mod.predict(X_test)
    mse = metrics.mean_squared_error(y_true=y_test, y_pred=y_hat)
    rsme = np.sqrt(mse)
    scores.append(rsme)

scores

# Para ver la media
np.mean(scores)

# Para ver la varianza
np.var(scores)
