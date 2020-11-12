# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:16:49 2020

@author: fhernanb
"""

# -------------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) para mostrar
# el uso de svm en regresion
# -------------------------------------------------------------------------

# Librerías a usar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm

# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/datos_svm_regresion.txt"
datos = pd.read_csv(file, comment='#', delimiter='\t')
datos.head()

# Explorando los datos
plt.scatter(x=datos.x, y=datos.y, color='purple', alpha=0.15)
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Diagrama de dispersión con 2 variables');

# Construyendo X e y
X = datos["x"]
y = datos["y"]

# El siguiente paso se debe hacer porque solo tenemos una covariable
# en el caso de varias no es necesario usar la siguiente instrucción.
X = X.values.reshape((-1, 1))

# Creando el modelo svm con kernel lineal
mod_lin = svm.SVR(C=1.0, epsilon=0.1, degree=1, kernel='linear')
mod_lin.fit(X, y)

# Para ver el modelo entrenado
mod_lin

# Para predecir Y cuando x1=3 y x1=4.2
mod_lin.predict([[3.0]])            # Uno por uno
mod_lin.predict([[4.2]])            # Uno por uno
mod_lin.predict([[3.0], [4.2]])     # Ambos

# Estimando y usando los datos de entrenamiento
y_hat_lin = mod_lin.predict(X)

# Funcion para calcular MSE
def mse(y, y_hat):
    return np.mean((y - y_hat_lin)**2)

mse(y, y_hat_lin)

# Calculando el coeficiente de correlacion de Pearson
np.corrcoef(y, y_hat_lin)

# Agregando el modelo estimado al diagrama originar
plt.scatter(x=datos.x, y=datos.y, color='purple', alpha=0.15)
plt.scatter(x=datos.x, y=y_hat_lin, color="red")
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Diagrama de dispersión con 2 variables')
plt.legend(('Y', 'Y estimado'))
plt.show()


