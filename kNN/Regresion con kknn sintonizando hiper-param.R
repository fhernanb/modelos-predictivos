# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar la base de datos Cars93
# del paquete MASS para estimar el precio del auto en funcion del
# peso y del rendimiento del combustible
# -------------------------------------------------------------------------

# Los datos que vamos a usar
library(MASS)
head(Cars93)

# Vamos a explorar los datos
library(tidyverse)
Cars93 %>% glimpse()

# Usando el paquete kknn --------------------------------------------------
library(kknn)

# train.kknn sirve para dos cosas:
# 1) para ajustar el modelo y
# 2) para encontrar los hiperparametros.
# En este ejemplo SIII vamos a sintonizar hiper-parametros
# Vamos a exigir que se pruebe con algunos kernel
# y maximo 70 vecinos.
# Nota: leave-one-out crossvalidation

fit1 <- train.kknn(Price ~ Weight + MPG.city,
                   data=Cars93,
                   kmax=70,
                   kernel = c("rectangular", "gaussian", 
                              "rank", "optimal"),
                   scale=TRUE)

# Para ver el modelo ajustado
fit1

# Para ver la clase del objeto
class(fit1)

# Para ver los elementos dentro del objeto
names(fit1)

# Para ver los valores de la sintonizacion
fit1$best.parameters

# Para ver los resultados en grafico
plot(fit1)

# -------------------------------------------------------------------------
# Tarea:
# Comparar los resultados de sintonizacion obtenidos con
# train.kknn con los de train usando el paquete caret.
# Se puede sintonizar q de la distancia de Minkowski?
# -------------------------------------------------------------------------

