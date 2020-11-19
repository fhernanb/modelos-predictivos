# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:29:45 2020

@author: fhernanb
"""

"""
La base de datos que vamos a usar en este ejemplo está disponible en el 
UCI Repository. El objetivo es crear una red neuronal para predecir 
la variable Y (target) definida como:
Y=1 (presence heart disease) si target es 1, 2, 3 o 4
Y=0 (absence heart disease) si target es 0
"""

# Librerías a usar
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# Leyendo los datos
file = "https://raw.githubusercontent.com/fhernanb/datos/master/cleveland.csv"
nombres = ('age', 'sex', 'cp', 'trestbps', 'chol',
           'fbs', 'restecg', 'thalach', 'exang', 
           'oldpeak', 'slope', 'ca', 'thal', 'target')
datos = pd.read_csv(file, comment='#', delimiter=',', names=nombres)
datos.head()

# Vamos a chequear si hay nan's
datos.isnull().values.any()
datos.isnull().sum().sum()

# Vamos a eliminar las filas con nan
datos = datos.dropna()

# Para explorar la variable respuesta
t1 = pd.crosstab(index=datos['target'], columns="Numero")
t1

# La variable respuesta es target que tiene cuatro números así: 
# target = 0 --> absence, 
# target = 1, 2, 3, 4 --> present. 
# Por esa razón vamos a modificar a target de la siguiente manera:

datos['targe'] = np.where(datos['target'] != 0, 1, 0)

# Creando X e y
y = datos["targe"]
X = datos.drop("target", axis=1)

# Para escalar los valores de X
scaledX = scale.fit_transform(X)

# Creando train y test
X_train, X_test, y_train, y_test = train_test_split(scaledX, y, 
                                                    test_size=0.20, 
                                                    random_state=42)

# Explorando la distribucion de y en train y test
pd.crosstab(index=y_train, columns="Numero")
pd.crosstab(index=y_test,  columns="Numero")

# Creando svm -----------------------------------------------------------------
mod = MLPClassifier(solver='adam', 
                    max_iter=1500,
                    alpha=1e-5,
                    activation='logistic',
                    hidden_layer_sizes=(14, 10, 5, 4), 
                    learning_rate='adaptive',
                    random_state=1)

mod.fit(X_train, y_train)

# Estimando y usando los datos de entrenamiento
y_hat = mod.predict(X_test)

# Matriz de confusión
confusion_matrix(y_test, y_hat)

# Para ver algunas medidas de desempeño
metrics.accuracy_score(y_test, y_hat)
metrics.balanced_accuracy_score(y_test, y_hat)

