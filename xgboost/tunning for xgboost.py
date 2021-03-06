# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:14:55 2020

@author: fhernanb
"""

"""
La primera parte fue tomada de:
    https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/
La parte de tunning fue tomada de:
    https://www.mikulskibartosz.name/xgboost-hyperparameter-tuning-in-python-using-grid-search/
"""

# evaluate xgboost algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

# Primera parte --------------------------------------------------------------

# define dataset
X, y = make_classification(n_samples=100, n_features=20, 
                           n_informative=15, n_redundant=5, 
                           random_state=7)

# define the model
model = XGBClassifier()

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', 
                           cv=cv, n_jobs=-1)

# report performance
print('Mean (and sd) of Accuracy: %.3f (%.3f)' % (mean(n_scores), 
                                         std(n_scores)))

# Segunda parte ---------------------------------------------------------------

# defining parameter range 
param_grid = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

# Creando el modelo
model = XGBClassifier(objective='binary:logistic',
                      nthread=4, seed=42)

grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    scoring = 'roc_auc',
                    n_jobs = 10,
                    cv = 10,
                    verbose=True)

# fitting the model for grid search 
grid.fit(X, y)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 

# Predictions with the best hyper-param combination
y_hat = grid.predict(X)







