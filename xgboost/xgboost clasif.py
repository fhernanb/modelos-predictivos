# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:27:19 2020

@author: fhernanb
"""

# make predictions using xgboost for classification
from numpy import mean
from numpy import std

from numpy import asarray
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)

# define the model
model = XGBClassifier()

# fit the model on the whole dataset
model.fit(X, y)

# make a single prediction
row = [0.2929949,-4.21223056,-1.288332,-2.17849815,-0.64527665,2.58097719,0.28422388,-7.1827928,-1.91211104,2.73729512,0.81395695,3.96973717,-2.66939799,3.34692332,4.19791821,0.99990998,-0.30201875,-4.43170633,-2.82646737,0.44916808]
row = asarray([row])
yhat = model.predict(row)
print('Predicted Class: %d' % yhat[0])

# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, 
                           scoring='accuracy',
                           cv=cv, n_jobs=-1, error_score='raise')

# to explore the results
n_scores

# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


