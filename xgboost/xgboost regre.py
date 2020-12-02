# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:19:26 2020

@author: fhernanb
"""

# evaluate xgboost ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)

# define the model
model = XGBRegressor()

# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, 
                           scoring='neg_mean_absolute_error',
                           cv=cv, n_jobs=-1, error_score='raise')

# to explore the results
n_scores

# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

