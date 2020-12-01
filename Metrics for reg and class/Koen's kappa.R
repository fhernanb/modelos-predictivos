# -------------------------------------------------------------------------
# En este script se muestra como calcular el Cohen's kappa coefficient
# -------------------------------------------------------------------------

library(psych)
library(MLmetrics)

# Para obtener ayuda sobre la funcion
help(cohen.kappa)

# Ejemplo 1 de Wikipedia --------------------------------------------------

# Creando la tabla como una matriz
tabla <- matrix(c(20, 5,
                  10, 15), byrow=TRUE, ncol=2)

# Para revisar la tabla
tabla

# Calculando
cohen.kappa(tabla)

# Ejemplo 2 de Wikipedia --------------------------------------------------

# Creando la tabla como una matriz
tabla <- matrix(c(45, 15,
                  25, 15), byrow=TRUE, ncol=2)

# Para revisar la tabla
tabla

# Calculando
cohen.kappa(tabla)

# Ejemplo 3 de Wikipedia --------------------------------------------------

# Creando la tabla como una matriz
tabla <- matrix(c(25, 35,
                   5, 35), byrow=TRUE, ncol=2)

# Para revisar la tabla
tabla

# Calculando
cohen.kappa(tabla)


# Ejemplo -----------------------------------------------------------------

fit <- glm(formula = vs ~ hp + wt,
              family=binomial(link="logit"), data=mtcars)

pred <- ifelse(fit$fitted.values < 0.5, 0, 1)
y <- mtcars$vs

# Matriz de confusión
tabla <- table(Prediccion=pred, Real=y)
tabla

# Metricas

# Accuracy manual
sum(diag(tabla)) / sum(tabla)

# Accuracy 
Accuracy(y_pred=pred, y_true=mtcars$vs)

# Cohen's kappa
cohen.kappa(cbind(y, pred))


# Ejemplo Naive^2 ---------------------------------------------------------
n <- length(y)

# Creando naive naive pred
pred <- sample(x=0:1, size=n, replace=TRUE)

# Matriz de confusión
tabla <- table(Prediccion=pred, Real=y)
tabla

# Accuracy 
Accuracy(y_pred=pred, y_true=mtcars$vs)
# Cohen's kappa
cohen.kappa(cbind(y, pred))

# Ejemplo Naive -----------------------------------------------------------
n <- length(y)

# Para ver la proporcion dentro de y
prop.table(table(y))

# Creando naive pred
pred <- sample(x=0:1, size=n, replace=TRUE, prob=c(0.5625, 0.4375))

# Matriz de confusión
tabla <- table(Prediccion=pred, Real=y)
tabla

# Accuracy 
Accuracy(y_pred=pred, y_true=mtcars$vs)
# Cohen's kappa
cohen.kappa(cbind(y, pred))


