# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:16:49 2020

@author: fhernanb
"""

"""
En este ejemplo se usan datos artificiales (simulados) para mostrar
como sintonizar hiper-parámetros en svm regresión
"""

# Librerías a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

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

# El siguiente paso se debe hacer porque solo tenemos una covariable
# en el caso de varias no es necesario usar la siguiente instrucción.
X = X.values.reshape((-1, 1))

# Para hacer tunning ---------------------------------------------------------

# defining parameter range 
param_grid = [
  {'C': [0.1, 0.5, 1, 1.5], 'kernel': ['linear']},
  {'C': [0.1, 0.5, 1, 1.5], 'degree': [2, 3, 4], 'kernel': ['poly']},
  {'C': [0.1, 0.5, 1, 1.5], 'gamma': [0.1, 0.5, 1, 1.5, 2], 'kernel': ['rbf']},
 ]

model = svm.SVR()
grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                    refit=True, verbose=3) 
  
# fitting the model for grid search 
grid.fit(X, y) 

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 

# Predictions with the best hyper-param combination
y_hat = grid.predict(X) 

# Para ver algunas medidas de desempeño
np.corrcoef(y, y_hat)

r2 = metrics.r2_score(y_true=y, y_pred=y_hat)
mse = metrics.mean_squared_error(y_true=y, y_pred=y_hat)

print("El valor de R2 es ", r2)
print("El valor de mse es ", mse)

# Agregando el modelo estimado al diagrama original
plt.scatter(x=datos.x, y=datos.y, color='black', alpha=0.55)
plt.plot(datos.x, y_hat, color='blue')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('SVM con sintonización')
plt.legend(('Modelo estimado', 'Y observado'))
plt.show()
