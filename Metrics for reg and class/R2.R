# -------------------------------------------------------------------------
# En este script se muestra como calcular R2
# -------------------------------------------------------------------------

# Ejemplo tomado de: 
# https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

y_true <- c(3, -0.5, 2, 7)
y_pred <- c(2.5, 0.0, 2, 8)

# Manualmente
1- sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)

# Usando una funcion
library(MLmetrics)
R2_Score(y_pred=y_pred, y_true=y_true)

