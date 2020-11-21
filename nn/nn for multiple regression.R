# -------------------------------------------------------------------------
# En este ejemplo se muestra como usar nn para regresion.
# El ejemplo esta basado en http://uc-r.github.io/ann_regression
# Los datos del ejemplo se van a simular pero tambien estan disponibles
# en un repo de github.
# -------------------------------------------------------------------------

# Simulando los datos -----------------------------------------------------
# Vamos a usar datos simulados de un modelo
# y ~ N(mu=4 - 3 * x1 + 3 * x2, sigma=6)

gen_dat <- function(n) {
  x1 <- runif(n=n, min=-5, max=6)
  x2 <- runif(n=n, min=-4, max=5)
  media <- 4 - 3 * x1 + 3 * x2
  y <- rnorm(n=n, mean=media, sd=6)
  marco_datos <- data.frame(y=y, x1=x1, x2=x2)
  return(marco_datos)
}

set.seed(1974)
datos <- gen_dat(n=100)
head(datos)

# Los datos simulados estan disponibles tambien en la url de abajo.
datos <- read.table("https://raw.githubusercontent.com/fhernanb/datos/master/datos_regresion_mult_redes.txt",
                    header=TRUE)
head(datos)


# Visualizando los datos --------------------------------------------------
library(scatterplot3d)
scatterplot3d(x=datos$x1, y=datos$x2, z=datos$y, 
              pch=16, cex.lab=1,
              highlight.3d=TRUE, type="h", xlab='x1',
              ylab='x2', zlab='y')


# Transformando los datos -------------------------------------------------

# Vamos a usar una transformacion al intervalo (0, 1).
# A usted le queda de tarea probar con una transformacion (-1, 1)

scale01 <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

library(dplyr)
datis <- datos %>% mutate_all(scale01) # scaled data

# Vamos a explorar la media y varianza de los datos sin/con transformacion
# pero vamos a crear una funcioncita para esto.
funcioncita <- function(x) c(Minimo=min(x), 
                             Media=mean(x), Mediana=median(x),
                             Desvi=sd(x), Vari=var(x), 
                             Maximo=max(x))

apply(datos, MARGIN=2, FUN=funcioncita) # sin transf
apply(datis, MARGIN=2, FUN=funcioncita) # con transf

# Ajustado el modelo con neuralnet ----------------------------------------

# Vamos a crear una red con 1 sola capa interna y 1 sola neurona
# funcion de activacion logistica

library(neuralnet)
mod1 <- neuralnet(y ~ x1 + x2, data=datis,
                  hidden=c(1),
                  rep=1,
                  algorithm="rprop+",
                  err.fct="sse",
                  act.fct="logistic")

# Dibujando la red entrenada
plot(mod1, rep = 'best')

# Para conocer la clase del objeto mod1
class(mod1)

# Para ver los objetos dentro de mod1
names(mod1)

# Explorando los pesos para luego hacer operaciones con ellos
mod1$weights

# Haciendo predicciones manuales para la observacion k-esima
k <- 5
datis[k, ] # primera linea

a <- mod1$weights[[1]][[1]][2, 1] * datis[k, 2] + 
  mod1$weights[[1]][[1]][3, 1] * datis[k, 3] +
  mod1$weights[[1]][[1]][1, 1]

b <- exp(a) / (1 + exp(a))

b * mod1$weights[[1]][[2]][2, 1] + mod1$weights[[1]][[2]][1, 1]

# Haciendo las predicciones automaticamente
predict(mod1, newdata=datis[k ,]) # igual al manual

# Creando un vector con todas las predicciones usando los datos transf
yhat1 <- predict(mod1, newdata=datis)

# Calculando el error
sum((datis$y - yhat1)^2) / 2

# Explorando las predicciones en el mundo transformado
par(mfrow=c(1, 2))

plot(x=datis$y, y=yhat1, las=1, xlab="y_t", 
     main="Transformed world")
abline(a=0, b=1, col="dodgerblue2", lwd=2)
cor(x=datis[, 1], y=yhat1)

# Explorando las predicciones en el mundo normal (no transf)
# Debemos usar la transformada inversa
yhat1_nt <- yhat1 * (max(datos$y) - min(datos$y)) + min(datos$y)
plot(x=datos$y, y=yhat1_nt, las=1, xlab="y",
     main="Real world")
abline(a=0, b=1, col="dodgerblue2", lwd=2)
cor(x=datos$y, y=yhat1_nt)

# Tarea: saque al menos UNA conclusion de este ejemplo.


# Ajustando el modelo con lm ----------------------------------------------

# Ahora vamos a ajustar el modelo usando lm 

mod_lm <- lm(y ~ x1 + x2, data=datos)
yhat_lm <- predict(mod_lm)

# Calculando el MSE
sum((datos$y - yhat_lm)^2) / 2 # Depende de las unidade de Y

# Comparando modelo nn y lm -----------------------------------------------
cor(x=datos$y, y=yhat1_nt)
cor(x=datos$y, y=yhat_lm)

par(mfrow=c(1, 2))

plot(x=datos$y, y=yhat1_nt, las=1, xlab="y", main="With nn")
abline(a=0, b=1, col="tomato", lwd=2)

plot(x=datos$y, y=yhat_lm, las=1, xlab="y", main="With lm")
abline(a=0, b=1, col="tomato", lwd=2)

par(mfrow=c(1, 1))

# Tarea: saque otra conclusion de este ejemplo.

# Variable importance -----------------------------------------------------

# Para ver la importancia de las variables en la red usamos
library(NeuralNetTools)

garson(mod1) # Garson (1991). Interpreting neural network connection weights
olden(mod1)  # Olden et al (2002). Illuminating the ’black-box’

# Tarea: Averiguar por que la altura de las barras son aprox 3 unidades

# Para ver la importancia de las variables en el modelo de regresion
summary(mod_lm)

# Para crear una figura similar a la de arriba pero para lm usamos
barplot(coef(mod_lm)[2:3], ylab="Importancia", las=1,
        ylim=c(-3.5, 3.5), col=c("navy", "deepskyblue"))
box()

# Ajustado el modelo con nnet ---------------------------------------------

# Vamos a crear una red con 1 sola capa interna y 1 sola neurona
# funcion de activacion logistica
# Nota: nnet solo permite UNA capa

library(nnet)
mod2 <- nnet(y ~ x1 + x2, data=datis,
             size=1,
             softmax=FALSE,
             maxit=1000)

# Dibujando la red entrenada
NeuralNetTools::plotnet(mod2)

# Para conocer la clase del objeto mod2
class(mod2)

# Para ver los objetos dentro de mod2
names(mod2)

# Para ver los pesos dentro de la red
mod2$wts

# Tarea: por que no son igualitos los pesos de ambas redes?

# Creando un vector con todas las predicciones usando los datos transf
yhat2 <- predict(mod2, newdata=datis)

# Calculando el error
sum((datis$y - yhat2)^2) / 2

# Explorando las predicciones con ambas redes
par(mfrow=c(1, 2))

plot(x=datis$y, y=yhat1, las=1, xlab="y_t", 
     main="Using neuralnet")

plot(x=datis$y, y=yhat2, las=1, xlab="y_t", 
     main="Using nnet")

# Tarea: Saque una conclusion del ejercicio.

# Para ver la importancia de las variables en la red usamos
library(NeuralNetTools)
garson(mod2) # Garson (1991). Interpreting neural network connection weights
olden(mod2)  # Olden et al (2002). Illuminating the ’black-box’


# Comparando los MSE ------------------------------------------------------

mse_neuralnet <- mean((datis$y - yhat1)^2)
mse_nn <- mean((datis$y - yhat2)^2)

cbind(mse_neuralnet, mse_nn)

# Tarea: volver a ajustar mod1 y mod2 pero modificando los otros 
# argumentos de las funciones y cambiando la ARQUITECTURA de la red
# para conseguir modelos con ERRORES menores a los mostrados aqui. 
# Le apuesto que usted logra disminuir aun mas los MSE.


# Explorando las utilidades del paquete NeuralNetTools --------------------

par(mfrow=c(1, 2))
plotnet(mod1)
plotnet(mod2, circle_col="tomato", bord_col="blue", prune_col="red")

