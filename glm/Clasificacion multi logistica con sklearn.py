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

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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

# Para definir el modelo
mod = LogisticRegression(penalty='l1',
                         fit_intercept=True,
                         #solver='liblinear',
                         max_iter=256)

# Para entrenar el modelo
mod.fit(X_train, y_train)

# Para ver el número de iteraciones
mod.n_iter_

# Para obtener los parametros
mod.intercept_
mod.coef_

# Para ver las probabilidades
proba = mod.predict_proba(X_test)
proba[0:5, ] # las primeras cinco probabilidades

# Para obtener las estimaciones
y_hat = mod.predict(X_test)
y_hat[0:5]

# Para crear la matriz de confusion
cm = metrics.confusion_matrix(y_test, y_hat)
print("La matriz de confusión obtenida es: \n")
print(cm)

# Para ver algunas medidas de desempeño
accu = metrics.accuracy_score(y_test, y_hat)
print("El valor de accuracy es ", accu)

