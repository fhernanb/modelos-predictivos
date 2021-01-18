# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 05:47:02 2020

@author: fhernanb
"""

"""
En este ejemplo vamos a utilizar la base de datos titanic
para predecir si una persona sobrevive o no usando como covariables
Pclass, Sex, Age y Fare.
"""

# Librerías a usar
import pandas as pd

from sklearn import metrics
from xgboost import XGBClassifier

# Los datos a usar estan disponibles en un repositorio de github
file = "https://raw.githubusercontent.com/fhernanb/datos/master/titanic.csv"
datos = pd.read_csv(file, sep=',')
datos.head()

# Vamos a seleccionar las variables de interes
datos = datos[["Survived",
               "Pclass",
               "Sex",
               "Age",
               "Fare"]]

# Explorando la variable respuesta
import seaborn as sns
sns.countplot(x='Survived', data=datos)

# Creando las variables dummy para las variables cualitativas
datos_dum = pd.get_dummies(datos, 
                           columns=['Pclass', 'Sex'], 
                           drop_first=True)

# ¿Cómo se ven los datos con variables dummy?
datos_dum.head

# Creando X e y
y = datos_dum["Survived"]
X = datos_dum.drop('Survived', axis=1)

# Creando el modelo de interés ------------------------------------------------

# Para definir el modelo
model = XGBClassifier()

# Para entrenar el modelo
model.fit(X, y)

# Estimando y usando los datos de train --------------
y_hat = model.predict(X)

# Confusion matrix
cm = metrics.confusion_matrix(y, y_hat)
print(cm)

# Para ver algunas medidas de desempeno
accu = metrics.accuracy_score(y_true=y, y_pred=y_hat)
print("El valor de accuracy es ", accu)
kappa = metrics.cohen_kappa_score(y, y_hat)
print("El valor de Kappa es ", kappa)




