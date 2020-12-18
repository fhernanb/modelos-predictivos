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

# Exploremos la relacion entre Y y las covariables
Cars93 %>% select(Price, Weight, MPG.city) %>% pairs()
# Tendra algun efecto los valores de x's?


# Ahora vamos a transformar las x's ---------------------------------------

Cars93 %>% mutate(weight_trans=Weight / sd(Weight),
                  mpgcity_trans=MPG.city / sd(MPG.city)) -> Cars93


# Usando el paquete kknn --------------------------------------------------
library(kknn)

# Modelo scale=FALSE
fit1 <- train.kknn(Price ~ Weight + MPG.city,
                   data=Cars93,
                   distance=3,
                   kmax=2,
                   kernel="gaussian",
                   scale=FALSE)

# Modelo scale=TRUE
fit2 <- train.kknn(Price ~ Weight + MPG.city,
                   data=Cars93,
                   distance=3,
                   kmax=2,
                   kernel="gaussian",
                   scale=TRUE)

# Modelo scale=FALSE pero con datos transformados
fit3 <- train.kknn(Price ~ weight_trans + mpgcity_trans,
                   data=Cars93,
                   distance=3,
                   kmax=2,
                   kernel="gaussian",
                   scale=FALSE)

# Vamos a comparar los modelos
fit1
fit2
fit3

# De la salida anterior vemos que fit2 y fit3 coinciden. 
# Por favor saque una conclusion de este ejemplo.
# Se debe transformar Y?

