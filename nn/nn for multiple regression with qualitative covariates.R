# -------------------------------------------------------------------------
# En este ejemplo se muestra como usar nn para regresion
# cuando se tienen covariables cualiativas.
# La idea es crear una red para predecir el Precio usando
# EngineSize y Type del auto.
# -------------------------------------------------------------------------

# Los datos ---------------------------------------------------------------
require(MASS)

# Para explorar los datos
library(dplyr)
Cars93 %>% glimpse()

# Contando NA's, hay varias formas de hacerlo, revisar el enlace de abajo
# https://sebastiansauer.github.io/sum-isna/
library(dplyr)
Cars93 %>%
  select(Price, EngineSize, Type) %>%  # replace to your needs
  summarise_all(funs(sum(is.na(.))))

# Creando la red sin transformar los datos --------------------------------
library(neuralnet)
mod0 <- neuralnet(Price ~ EngineSize + Type, data=Cars93,
                  rep=10,
                  algorithm="rprop+",
                  err.fct="sse",
                  act.fct="logistic",
                  hidden=c(2, 2))       # Nos va a salir un error?

# Por que apararecio ese error?

# Transformando las variables CUALITATIVAS a 0 y 1 ------------------------
library(caret)

# usando model.matrix
mt_dummies <- model.matrix(Price ~ EngineSize + Type, data=Cars93)
head(mt_dummies)

# usando dummyVars
dummies <- dummyVars(Price ~ EngineSize + Type, data=Cars93)
mt_dummies <- predict(dummies, newdata=Cars93)
head(mt_dummies)

datos <- cbind(Price=Cars93$Price, mt_dummies)
datos <- as.data.frame(datos)
head(datos)

# Transformando los datos -------------------------------------------------

preProcValues <- preProcess(datos, method=c("range"))
datis <- predict(preProcValues, datos)
head(datis)

# Vamos a explorar la media y varianza de los datos sin/con transformacion
# pero vamos a crear una funcioncita para esto.
funcioncita <- function(x) c(min=min(x), med=mean(x), max=max(x))

apply(datos, MARGIN=2, FUN=funcioncita) # sin transf
apply(datis, MARGIN=2, FUN=funcioncita) # con transf

# Ajustado el modelo con neuralnet ----------------------------------------
library(neuralnet)
mod1 <- neuralnet(Price ~ ., data=datis,
                  rep=10,
                  algorithm="rprop+",
                  err.fct="sse",
                  act.fct="logistic",
                  hidden=c(5, 2))

# Dibujando la red entrenada
plot(mod1, rep='best')

# Haciendo las predicciones automaticamente
y_hat_t <- predict(mod1, newdata=datis)

# Calculando el error
sum((datis$Price - y_hat_t)^2) / 2

# Explorando las predicciones en el mundo transformado
par(mfrow=c(1, 2))

plot(x=datis$Price, y=y_hat_t, las=1, xlab="y_t", 
     main="Transformed world")
abline(a=0, b=1, col="dodgerblue2", lwd=2)
cor(x=datis$Price, y=y_hat_t)

# Explorando las predicciones en el mundo normal (no transf)
y_hat <- y_hat_t * (max(datos$Price) - min(datos$Price)) + min(datos$Price)

plot(x=datos$Price, y=y_hat, las=1, xlab="y",
     main="Real world")
abline(a=0, b=1, col="dodgerblue2", lwd=2)

# Calculando mse
mse <- function(y, y_hat) mean((y_hat - y)^2)
mse(datos$Price, y_hat)

# Calculando la correlacion
cor(x=datos$Price, y=y_hat)

# Variable importance -----------------------------------------------------
library(NeuralNetTools)
garson(mod1)
olden(mod1)


