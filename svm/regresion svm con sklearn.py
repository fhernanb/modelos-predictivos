# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:16:49 2020

@author: fhernanb
"""

"""
En este ejemplo se usan datos artificiales (simulados) para mostrar
el uso de svm en regresion.
"""

# Librerías a usar
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics

# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/datos_svm_regresion.txt"
datos = pd.read_csv(file, comment='#', delimiter='\t')
datos.head()

# Explorando los datos
plt.scatter(x=datos.x, y=datos.y, color='black', alpha=0.55)
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Diagrama de dispersión con 2 variables');

# Construyendo X e y
X = datos["x"]
y = datos["y"]

# El siguiente paso se debe hacer porque solo tenemos UNA covariable,
# en el caso de varias no es necesario usar la siguiente instrucción.
X = X.values.reshape((-1, 1))

# Creando el modelo svm con kernel lineal ---------------------------------

# Para detalles sobre la función que vamos a usar visite:
import webbrowser
webbrowser.open('https://tinyurl.com/y3aa67tn')

mod_lin = svm.SVR(C=1.0, epsilon=0.1, degree=1, kernel='linear')
mod_lin.fit(X, y)

# Para predecir Y cuando x1=3 y x1=4.2
mod_lin.predict([[3.0]])            # Uno por uno
mod_lin.predict([[4.2]])            # Uno por uno
mod_lin.predict([[3.0], [4.2]])     # Ambos

# Estimando y usando los datos de entrenamiento
y_hat_lin = mod_lin.predict(X)

# Para ver algunas medidas de desempeño
metrics.r2_score(y_true=y, y_pred=y_hat_lin)
metrics.mean_squared_error(y_true=y, y_pred=y_hat_lin)

# Agregando el modelo estimado al diagrama originar
plt.scatter(x=datos.x, y=datos.y, color='black', alpha=0.55)
plt.plot(datos.x, y_hat_lin, color='red')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('SVM con kernel lineal')
plt.legend(('Modelo estimado', 'Y observado'))
plt.show()


# Creando el modelo svm con kernel polinomial -------------------------------

mod_pol = svm.SVR(C=1.0, epsilon=0.1, degree=1, coef0=0, gamma=1, kernel='poly')
mod_pol.fit(X, y)

# Para ver el modelo entrenado
mod_pol

# Para predecir Y cuando x1=3 y x1=4.2
mod_pol.predict([[3.0]])            # Uno por uno
mod_pol.predict([[4.2]])            # Uno por uno
mod_pol.predict([[3.0], [4.2]])     # Ambos

# Estimando y usando los datos de entrenamiento
y_hat_pol = mod_pol.predict(X)

# Para ver algunas medidas de desempeño
metrics.r2_score(y_true=y, y_pred=y_hat_pol)
metrics.mean_squared_error(y_true=y, y_pred=y_hat_pol)

# Agregando el modelo estimado al diagrama originar
plt.scatter(x=datos.x, y=datos.y, color='black', alpha=0.55)
plt.plot(datos.x, y_hat_pol, color='blue')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('SVM con kernel lineal')
plt.legend(('Modelo estimado', 'Y observado'))
plt.show()


# Creando el modelo svm con kernel radial  -------------------------------

mod_rad = svm.SVR(C=1.0, epsilon=0.1, gamma=1, kernel='rbf')
mod_rad.fit(X, y)

# Para ver el modelo entrenado
mod_rad

# Para predecir Y cuando x1=3 y x1=4.2
mod_rad.predict([[3.0]])            # Uno por uno
mod_rad.predict([[4.2]])            # Uno por uno
mod_rad.predict([[3.0], [4.2]])     # Ambos

# Estimando y usando los datos de entrenamiento
y_hat_rad = mod_rad.predict(X)

# Para ver algunas medidas de desempeño
metrics.r2_score(y_true=y, y_pred=y_hat_rad)
metrics.mean_squared_error(y_true=y, y_pred=y_hat_rad)

# Agregando el modelo estimado al diagrama originar
plt.scatter(x=datos.x, y=datos.y, color='black', alpha=0.55)
plt.plot(datos.x, y_hat_rad, color='green')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('SVM con kernel lineal')
plt.legend(('Modelo estimado', 'Y observado'))
plt.show()


# Comparando los tres modelos --------------------------------------------

plt.scatter(x=datos.x, y=datos.y, color='black', alpha=0.55)
plt.plot(datos.x, y_hat_lin, color='red', linewidth=5)
plt.plot(datos.x, y_hat_pol, color='blue')
plt.plot(datos.x, y_hat_rad, color='green')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('SVM con kernel lineal')
plt.legend(('SVM lin', 'SVM poly', 'SVM rad', 'Y observado'))
plt.show()

