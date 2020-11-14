# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:00:49 2020

@author: fhernanb
"""

"""
El ejemplo mostrado aquí fue tomado y adaptado de la publicación
How to Tune Algorithm Parameters with Scikit-Learn
de Jason Brownlee disponible en https://tinyurl.com/y6zfg3r3
"""

# Grid Search Parameter Tuning -----------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# load the diabetes datasets
dataset = datasets.load_diabetes()

# to write explicitly X and y
X = dataset.data
y = dataset.target

# prepare a range of alpha values to test
alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])

# create and fit a ridge regression model, testing each alpha
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(dataset.data, dataset.target)
print(grid)

# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)



