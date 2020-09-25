# En este ejemplo se muestra como usar nn para regresion
# cuando se tienen covariables cualiativas


# Los datos ---------------------------------------------------------------

require(MASS)
str(Cars93)
Cars93$Type <- relevel(Cars93$Type, ref = 'Small')
levels(Cars93$Type)  # Para verificar el cambio


# Creando la red sin transformar los datos --------------------------------
library(neuralnet)
mod0 <- neuralnet(Price ~ EngineSize + Type, data=Cars93,
                  rep=10,
                  algorithm="rprop+",
                  err.fct="sse",
                  act.fct="logistic",
                  hidden=c(1))


# Transformando las variables CUALITATIVAS a 0 y 1 ------------------------
library(caret)

# usando model.matrix
mt_dummies <- model.matrix(Price ~ EngineSize + Type, data = Cars93)
head(mt_dummies)

# usando dummyVars
dummies <- dummyVars(Price ~ EngineSize + Type, data = Cars93)
mt_dummies <- predict(dummies, newdata = Cars93)
head(mt_dummies)

datos <- cbind(Price=Cars93$Price, mt_dummies)
datos <- as.data.frame(datos)
head(datos)

# Transoformando los datos ------------------------------------------------

preProcValues <- preProcess(datos, method = c("range"))
datis <- predict(preProcValues, datos)
head(datis)

# Ajustado el modelo con neuralnet ----------------------------------------
library(neuralnet)
mod1 <- neuralnet(Price ~ ., data=datis,
                  rep=10,
                  algorithm="rprop+",
                  err.fct="sse",
                  act.fct="logistic",
                  hidden=c(5, 2))

# Dibujando la red entrenada
plot(mod1, rep = 'best')

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
cor(x=datos$Price, y=y_hat)

# Ajustando el modelo con lm ----------------------------------------------
mod2 <- lm(Price ~ EngineSize + Type, data=Cars93)
y_hat_lm <- predict(mod2)

# Comparando modelo nn y lm -----------------------------------------------
mse <- function(y, y_hat) mean((y_hat - y)^2)

cor(x=datos$Price, y=y_hat)
cor(x=datos$Price, y=y_hat_lm)

mse(datos$Price, y_hat)
mse(datos$Price, y_hat_lm)

par(mfrow=c(1, 2))

plot(x=datos$Price, y=y_hat, las=1, xlab="y", main="With nn")
abline(a=0, b=1, col="tomato", lwd=2)

plot(x=datos$Price, y=y_hat_lm, las=1, xlab="y", main="With lm")
abline(a=0, b=1, col="tomato", lwd=2)

# Variable importance -----------------------------------------------------
library(NeuralNetTools)
garson(mod1)
olden(mod1)

