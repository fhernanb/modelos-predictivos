# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:27:17 2020

@author: fhern
"""

"""
En este ejemplo vamos a utilizar la base de datos titanic
para predecir si una persona sobrevive o no usando como covariables
Pclass, Sex, Age y Fare.
"""

# Librerías a usar
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Creando train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=123)

# Creando el modelo de interés ------------------------------------------------

# Para definir el modelo
mod = LogisticRegression(fit_intercept=True,
                         solver='liblinear',
                         max_iter=170)

# Para entrenar el modelo
mod.fit(X_train, y_train)

# Para ver el número de iteraciones
mod.n_iter_

# Para obtener los parametros
mod.intercept_
mod.coef_

# Para ver las probabilidades
proba = mod.predict_proba(X_train)
proba[0:5, ] # las primeras cinco probabilidades

# Para obtener las estimaciones
y_hat = mod.predict(X_test)
y_hat[0:5]

# Utilizando statmodels ---------------------------------------------
import statsmodels.api as sm

# Vamos a agregar la columna de 1 al inicio para intercepto
X_train = sm.add_constant(X_train)

mod2 = sm.Logit(y_train, X_train)
result = mod2.fit()

# printing the summary table 
result.summary()

# Para ver los coeficientes
coefficients = result.params
coefficients



