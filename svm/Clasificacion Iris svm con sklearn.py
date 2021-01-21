# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:27:17 2020

@author: fhernanb
"""

"""
En este ejemplo vamos a utilizar la base de datos clásica 
llamada Iris. El objetivo es crear un modelo de clasificacion
para clasificar nuevas flores en Setosa, Virginica o Versicolor 
en función de dos covariables: Sepal.Width y Sepal.Length.
"""

# Librerías a usar
import pandas as pd
import numpy as np

from sklearn import svm

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

scale = StandardScaler()

# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/iris.txt"
datos = pd.read_csv(file, sep='\t')
datos.head()

# Vamos a seleccionar las variables de interes
datos = datos[["Species",
               "Sepal.Width",
               "Sepal.Length"]]

# Vamos a convertir la variable Species a numérica
enc = OrdinalEncoder()
enc.fit(datos[["Species"]])
datos["y"] = enc.transform(datos[["Species"]])

# Explorando la variable respuesta
import seaborn as sns
sns.countplot(x='Species', data=datos)

# Creando X e y
y = datos["y"]
X = datos[["Sepal.Length", "Sepal.Width"]]

# Para escalar los valores de X
scaledX = scale.fit_transform(X)

# Creando train y test
X_train, X_test, y_train, y_test = train_test_split(scaledX, y, 
                                                    test_size=0.20, 
                                                    random_state=42)

# Utilizando sklearn -----------------------------------------------

# SVM con kernel lineal --------

# Para definir el modelo
mod_lin = svm.SVC(kernel='linear')

# Para entrenar el modelo
mod_lin.fit(X_train, y_train)

# Para obtener las estimaciones
y_hat = mod_lin.predict(X_test)
y_hat[0:5]

# Para crear la matriz de confusion
cm = metrics.confusion_matrix(y_test, y_hat)
print("La matriz de confusión obtenida es: \n")
print(cm)

# Para ver algunas medidas de desempeño
accu = metrics.accuracy_score(y_test, y_hat)
print("El valor de accuracy es ", accu)

# Para dibujar las fronteras ----------------------------------

import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Set-up grid for plotting.
X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots()
plot_contours(ax, mod_lin, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
ax.set_xlabel('Sepal.Length')
ax.set_ylabel('Sepal.Width')
ax.set_title('Regiones kernel lineal')
plt.show()

# SVM con kernel poly --------

# Para definir el modelo
mod_poly = svm.SVC(kernel='poly')

# Para entrenar el modelo
mod_poly.fit(X_train, y_train)

# Para obtener las estimaciones
y_hat = mod_poly.predict(X_test)
y_hat[0:5]

# Para crear la matriz de confusion
cm = metrics.confusion_matrix(y_test, y_hat)
print("La matriz de confusión obtenida es: \n")
print(cm)

# Para ver algunas medidas de desempeño
accu = metrics.accuracy_score(y_test, y_hat)
print("El valor de accuracy es ", accu)

# Para dibujar las fronteras ----------------------------------

# Set-up grid for plotting.
X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots()
plot_contours(ax, mod_poly, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
ax.set_xlabel('Sepal.Length')
ax.set_ylabel('Sepal.Width')
ax.set_title('Regiones kernel poly')
plt.show()

# SVM con kernel rbf --------

# Para definir el modelo
mod_rbf = svm.SVC(kernel='rbf')

# Para entrenar el modelo
mod_rbf.fit(X_train, y_train)

# Para obtener las estimaciones
y_hat = mod_rbf.predict(X_test)
y_hat[0:5]

# Para crear la matriz de confusion
cm = metrics.confusion_matrix(y_test, y_hat)
print("La matriz de confusión obtenida es: \n")
print(cm)

# Para ver algunas medidas de desempeño
accu = metrics.accuracy_score(y_test, y_hat)
print("El valor de accuracy es ", accu)

# Para dibujar las fronteras ----------------------------------

# Set-up grid for plotting.
X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots()
plot_contours(ax, mod_rbf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
ax.set_xlabel('Sepal.Length')
ax.set_ylabel('Sepal.Width')
ax.set_title('Regiones kernel rbf')
plt.show()

# SVM con kernel sigmoid --------

# Para definir el modelo
mod_sig = svm.SVC(kernel='sigmoid')

# Para entrenar el modelo
mod_sig.fit(X_train, y_train)

# Para obtener las estimaciones
y_hat = mod_sig.predict(X_test)
y_hat[0:5]

# Para crear la matriz de confusion
cm = metrics.confusion_matrix(y_test, y_hat)
print("La matriz de confusión obtenida es: \n")
print(cm)

# Para ver algunas medidas de desempeño
accu = metrics.accuracy_score(y_test, y_hat)
print("El valor de accuracy es ", accu)

# Para dibujar las fronteras ----------------------------------

# Set-up grid for plotting.
X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots()
plot_contours(ax, mod_sig, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
ax.set_xlabel('Sepal.Length')
ax.set_ylabel('Sepal.Width')
ax.set_title('Regiones kernel sigmoid')
plt.show()

# Para hacer tunning ---------------------------------------------------------

# defining parameter range 
param_grid = [
  {'C': [0.1, 0.5, 1, 1.5], 'kernel': ['linear']},
  {'C': [0.1, 0.5, 1, 1.5], 'degree': [2, 3, 4], 'kernel': ['poly']},
  {'C': [0.1, 0.5, 1, 1.5], 'gamma': [0.1, 0.5, 1], 'kernel': ['rbf']},
  {'C': [0.1, 0.5, 1, 1.5], 'gamma': [0.1, 0.5, 1], 'kernel': ['sigmoid']}
 ]

model = svm.SVC()
grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                    refit=True, verbose=3) 

# fitting the model for grid search 
grid.fit(X_train, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 

# Predictions with the best hyper-param combination
y_hat = grid.predict(X_test) 

# Para crear la matriz de confusion
cm = metrics.confusion_matrix(y_test, y_hat)
print("La matriz de confusión obtenida es: \n")
print(cm)

# Para ver algunas medidas de desempeño
accu = metrics.accuracy_score(y_test, y_hat)
print("El valor de accuracy es ", accu)

# Para dibujar las fronteras ----------------------------------

# Set-up grid for plotting.
X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots()
plot_contours(ax, grid, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
ax.set_xlabel('Sepal.Length')
ax.set_ylabel('Sepal.Width')
ax.set_title('Regiones para modelo sintonizado')
plt.show()

