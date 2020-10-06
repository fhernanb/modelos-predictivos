# -------------------------------------------------------------------------
# En este ejemplo se muestra como usar la transformacion inversa para
# obtener las predicciones en la escala original
# -------------------------------------------------------------------------

# Los datos que vamos a usar
library(MASS)
head(Cars93)

# Creando la base de datos solo con las variables de interes
library(dplyr)
datos <- Cars93 %>% select(Price, Weight, MPG.city)

# Curioseando los datos
pairs(datos, las=1, col="tomato")

# Pre procesando los datos al intervalo 0-1
library(caret)
preProcValues <- preProcess(datos, method="range")
datos_transf <- predict(preProcValues, datos)

# Ajustando el modelo con los datos transformados
fit1 <- lm(Price ~ Weight + MPG.city, data=datos_transf)
y_hat1 <- predict(fit1, newdata=datos_transf)
y_hat1 <- y_hat1 * (max(datos$Price) - min(datos$Price)) + min(datos$Price)

# Ajustando el modelo con los datos originales
fit2 <- lm(Price ~ Weight + MPG.city, data=datos)
y_hat2 <- predict(fit2, newdata=datos)

# Comparando
plot(x=y_hat1, y=y_hat2, las=1, pch=20)


