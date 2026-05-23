# -------------------------------------------------------------------------
# En este ejemplo se muestra como usar nn para regresion.
# El ejemplo esta basado en http://uc-r.github.io/ann_regression
# Los datos del ejemplo se van a simular pero tambien estan disponibles
# en un repositorio de github.
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

# Los datos simulados estan disponibles tambien en la url de abajo:
url <- "https://raw.githubusercontent.com/fhernanb/datos/master/datos_regresion_mult_redes.txt"
datos <- read.table(url, header=TRUE)
head(datos)

# Visualizando los datos --------------------------------------------------
library(scatterplot3d)
scatterplot3d(x=datos$x1, y=datos$x2, z=datos$y, 
              pch=16, cex.lab=1,
              highlight.3d=TRUE, type="h", xlab="x1",
              ylab="x2", zlab="y")

# Transformando los datos -------------------------------------------------

# Vamos a usar una transformacion al intervalo (0, 1) para
# transformar todas las variables Y y X's simultaneamente.
# La formula para transformar a (0, 1) es:
# (x - min(x)) / (max(x) - min(x))
# y la formula inversa es: 
# x * (max(x) - min(x)) + min(x)

# A usted le queda de tarea probar con una transformacion (-1, 1)

library(recipes)

# 1. Creando mi receta (recipe) para transformar
my_recipe <- recipe(y ~ x1 + x2, data=datos) |>
  step_range(all_numeric(), min=0, max=1)

# 2. Preparando la recepta con prep()
trained_recipe <- prep(my_recipe, training=datos)

# 3. Horneando la receta con bake() para obtener los datos escalados
datos_transf <- bake(trained_recipe, new_data=datos)
datos_transf

# Vamos a explorar la media y varianza de los datos sin/con transformacion
# pero vamos a crear una funcioncita para esto.
funcioncita <- function(x) c(Minimo=min(x), 
                             Media=mean(x), 
                             Mediana=median(x),
                             Desvi=sd(x), 
                             Vari=var(x), 
                             Maximo=max(x))

apply(datos,        MARGIN=2, FUN=funcioncita) # sin transformar
apply(datos_transf, MARGIN=2, FUN=funcioncita) # con transformacion

# Ajustado el modelo con neuralnet ---------------------------------------------

# Vamos a crear una red con 1 sola capa interna y 1 sola neurona
# funcion de activacion logistica y otras caracteristica

library(neuralnet)
set.seed(1267)
mod1 <- neuralnet(y ~ x1 + x2, data=datos_transf,
                  hidden=c(1),
                  rep=1,
                  algorithm="rprop+",
                  err.fct="sse",
                  act.fct="logistic")

# Dibujando la red entrenada
plot(mod1, rep = "best")

# Para conocer la clase del objeto mod1
class(mod1)

# Para ver los objetos dentro de mod1
names(mod1)

# Explorando los pesos para luego hacer operaciones con ellos
mod1$weights

# Haciendo predicciones manuales para la observacion k-esima
k <- 5
datos_transf[k, ] # primera linea

a <- mod1$weights[[1]][[1]][2, 1] * datos_transf[k, "x1"] + 
  mod1$weights[[1]][[1]][3, 1] * datos_transf[k, "x2"] +
  mod1$weights[[1]][[1]][1, 1]

a

# Como la red neuronal se creo con funcion de activacion logistica
# yo debo usar la inversa de ella para obtener los resultados, como en glm binom
# f(x) = exp(x) / (1 + exp(x))
# La funcion la tome de https://en.wikipedia.org/wiki/Logit

b <- exp(a) / (1 + exp(a))

b

b * mod1$weights[[1]][[2]][2, 1] + mod1$weights[[1]][[2]][1, 1]

# Haciendo las predicciones automaticamente
predict(mod1, newdata=datos_transf[k ,]) # igual al manual

# Creando un vector con todas las predicciones usando los datos transf
yhat1_transf <- predict(mod1, newdata=datos_transf)
yhat1_transf <- as.vector(yhat1_transf) # para ver como vector

# Explorando las predicciones en el mundo transformado
par(mfrow=c(1, 2), mai=c(1, 1, 1, 1) + 0.1)

plot(x=datos_transf$y, y=yhat1_transf, las=1, xlab="y_transf", 
     main="Transformed world")
abline(a=0, b=1, col="dodgerblue2", lwd=2)

# Correlacion entre y and y_hat
cor(x=datos_transf$y, y=yhat1_transf)

# Explorando las predicciones en el mundo normal (no transf)
# Debemos usar la transformada inversa
yhat1 <- yhat1_transf * (max(datos$y) - min(datos$y)) + min(datos$y)

plot(x=datos$y, y=yhat1, las=1, xlab="y",
     main="Real world")
abline(a=0, b=1, col="tomato", lwd=2)

# Correlacion entre y and y_hat
cor(x=datos$y, y=yhat1)

# Tarea: saque al menos UNA conclusion de este ejemplo.

# Ajustando el modelo con lm ---------------------------------------------------

# Ahora vamos a ajustar el modelo usando lm solo para poder
# comparar los resultados de nn con lm

mod_lm <- lm(y ~ x1 + x2, data=datos)
yhat_lm <- predict(mod_lm)

# Calculando el MSE
library(yardstick)
mse_vec(truth=datos$y, estimate=yhat_lm)
mse_vec(truth=datos$y, estimate=yhat1)

# Cual modelo presenta el menor MSE?

# Vamos a comparar graficamente las estimaciones con nn y lm
par(mfrow=c(1, 2))

plot(x=datos$y, y=yhat1, las=1, xlab="y", main="With nn")
abline(a=0, b=1, col="darkgreen", lwd=2)

plot(x=datos$y, y=yhat_lm, las=1, xlab="y", main="With lm")
abline(a=0, b=1, col="orange", lwd=2)

par(mfrow=c(1, 1))

cor(x=datos$y, y=yhat1)
cor(x=datos$y, y=yhat_lm)

# Tarea: saque otra conclusion de este ejemplo.

# ------------------------------------------------------------------------------
# Como obtener predicciones de Y para nuevos casos usando nn???

# Supongamos que queremos estimar Y para tres nuevos casos:
# caso 1: x1=-2, x2=4
# caso 2: x1= 0, x2=3
# caso 3: x1= 1, x2=1

# Para hacer la estimacion hacemos lo siguiente:

# Creamos un nuevo dataframe con los nuevos datos, como no 
# conocemos los valores de Y (obvio) le colocamos NA en la columna Y.
new_data <- data.frame(x1=c(-2, 0, 1),
                       x2=c(4, 3, 1),
                       y=NA)

new_data

# Transformar los datos con la informacion almacenada en el objeto trained_recipe
new_data_transf <- bake(trained_recipe, new_data=new_data)

# Comparar los datos transformar y transformados
new_data
new_data_transf

# Hacer la prediccion
yhat_new <- predict(mod1, newdata=new_data_transf)

# Transformar la prediccion a la escala original de Y
yhat_new <- yhat_new * (max(datos$y) - min(datos$y)) + min(datos$y)

# Agregando la prediccion al dataframe nuevo
new_data$y <- yhat_new
new_data


# Variable importance -----------------------------------------------------

# Para ver la importancia de las variables en la red usaremos el 
# paquete NeuralNetTools y aplicamos al objeto mod1 que es de clase nn

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
set.seed(2583)
mod2 <- nnet(y ~ x1 + x2, data=datos_transf,
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
yhat2_transf <- predict(mod2, newdata=datos_transf)
yhat2_transf <- as.vector(yhat2_transf) # para ver como vector
yhat2 <- yhat2_transf * (max(datos$y) - min(datos$y)) + min(datos$y)

# Explorando las predicciones con ambas redes
par(mfrow=c(1, 2))

plot(x=datos$y, y=yhat1, las=1, xlab="y", 
     main="Using neuralnet")

plot(x=datos$y, y=yhat2, las=1, xlab="y", 
     main="Using nnet")

# Calculando el MSE
library(yardstick)
mse_vec(truth=datos$y, estimate=yhat1)
mse_vec(truth=datos$y, estimate=yhat2)

# Tarea: Saque una conclusion del ejercicio.

# Para ver la importancia de las variables en la red usamos
library(NeuralNetTools)
garson(mod2) # Garson (1991). Interpreting neural network connection weights
olden(mod2)  # Olden et al (2002). Illuminating the "black-box"

# Nota: ohhh, de esa manera podemos saber si una X es importante y que tan imp.

# Tarea: volver a ajustar mod1 y mod2 pero modificando los otros 
# argumentos de las funciones y cambiando la ARQUITECTURA de la red
# para conseguir modelos con MSE menores a los mostrados aqui. 
# Le apuesto que usted logra disminuir aun mas los MSE.


# Explorando las utilidades del paquete NeuralNetTools --------------------

par(mfrow=c(1, 2))
plotnet(mod1)
plotnet(mod2)
