# -------------------------------------------------------------------------
# En este ejemplo se usa la base de datos Auto del paquete ISLR.
# Los datos estan disponibles tambien en la url:
# https://raw.githubusercontent.com/fhernanb/datos/master/Wage_ISLR.txt
#
# La idea es crear un modelo para predecir el estado marital de trabajadores
# usando informacion de la base de datos
# -------------------------------------------------------------------------

library(ISLR)

# Vamos a explorar los datos
library(tidyverse)
Wage %>% glimpse()

# Los mismos datos estan disponibles en 
# https://raw.githubusercontent.com/fhernanb/datos/master/Wage_ISLR.txt
file <- "https://raw.githubusercontent.com/fhernanb/datos/master/Wage_ISLR.txt"
datos <- read.table(file, header=TRUE)
head(datos)

# Pasando a factor la respuesta
datos$maritl <- factor(datos$maritl)
y <- datos$maritl

# Explorando los datos
library(ggplot2)

# Relacion entre maritl y education
tabla <- with(datos, table(maritl, education))
tabla <- prop.table(tabla, margin=2)
df <- as.data.frame(tabla)
ggplot(df, aes(x = education, y = Freq, fill = maritl)) +
  geom_col(position = "dodge")

# Relacion entre maritl y health
tabla <- with(datos, table(maritl, health))
tabla <- prop.table(tabla, margin=2)
df <- as.data.frame(tabla)
ggplot(df, aes(x = health, y = Freq, fill = maritl)) +
  geom_col(position = "dodge")

# Relacion entre maritl y race
tabla <- with(datos, table(maritl, race))
tabla <- prop.table(tabla, margin=2)
df <- as.data.frame(tabla)
ggplot(df, aes(x = race, y = Freq, fill = maritl)) +
  geom_col(position = "dodge")

# Relacion entre maritl y jobclass
tabla <- with(datos, table(maritl, jobclass))
tabla <- prop.table(tabla, margin=2)
df <- as.data.frame(tabla)
ggplot(df, aes(x = jobclass, y = Freq, fill = maritl)) +
  geom_col(position = "dodge")

# svm lineal --------------------------------------------------------------
library(kernlab)

# Para ajustar el modelo
mod_lin <- ksvm(maritl ~ age + education + health + race + jobclass, 
                data=datos,
                type="C-svc", 
                kernel="vanilladot",
                C=1, epsilon=0.1)

# To obtain y_hat
y_hat_lin <- predict(mod_lin)

# To illustrate the results
tabla <- table(predict=y_hat_lin, Group=y)
tabla

# To obtain the metrics
MLmetrics::Accuracy(y_pred=y_hat_lin, y_true=y)

# svm polinomial ----------------------------------------------------------

# Para ajustar el modelo con los hiper-parametros por defecto
mod_pol <- ksvm(maritl ~ age + education + health + race + jobclass, 
                data=datos,
                type="C-svc",
                kernel="polydot",
                C=1, epsilon=0.1, 
                kpar=list(degree=1, scale=1, offset=1))

# To obtain y_hat
y_hat_pol <- predict(mod_pol)

# To illustrate the results
tabla <- table(predict=y_hat_pol, Group=y)
tabla

# To obtain the metrics
MLmetrics::Accuracy(y_pred=y_hat_pol, y_true=y)


# svm radial --------------------------------------------------------------

# Para ajustar el modelo con los hiper-parametros por defecto
mod_rad <- ksvm(maritl ~ age + education + health + race + jobclass, 
                data=datos,
                type="C-svc",
                kernel="rbfdot",
                C=1, epsilon=0.1,
                kpar=list(sigma=1))

# To obtain y_hat
y_hat_rad <- predict(mod_rad)

# To illustrate the results
tabla <- table(predict=y_hat_rad, Group=y)
tabla

# To obtain the metrics
MLmetrics::Accuracy(y_pred=y_hat_rad, y_true=y)

# Comparing ---------------------------------------------------------------
MLmetrics::Accuracy(y_pred=y_hat_lin, y_true=y)
MLmetrics::Accuracy(y_pred=y_hat_pol, y_true=y)
MLmetrics::Accuracy(y_pred=y_hat_rad, y_true=y)

