# -------------------------------------------------------------------------
# En este ejemplo se muestra como crear una red neurnal para clasificar
# objetos en dos grupos
# -------------------------------------------------------------------------



# Los datos ---------------------------------------------------------------

# Vamos a usar un paquete que esta alojado en github y que permite
# crear datos agrupados (cluster)
if (!require('devtools')) install.packages('devtools')
devtools::install_github('elbamos/clusteringdatasets', force=TRUE)

# Para cargar el paquete
library(clusteringdatasets)

# Para crear los datos
moons <- make_moons(n_samples=100, shuffle=FALSE, noise=0.25)

# Vamos a crear dos grupos pero los vamos a llamar por colores
y <- ifelse(moons$labels == 1, "grupo_1", "grupo_2")
y <- as.factor(y)

# Para dibujar los datos simulados
plot(moons$samples, pch=20, las=1, col=c("tomato", "dodgerblue")[y])

# Vamos a transformar los datos a un dataframe
datos <- data.frame(y=y,
                    x1=moons$samples[, 1],
                    x2=moons$samples[, 2])

head(datos)


# Ajustado el modelo con neuralnet ----------------------------------------

# Vamos a crear una red con 1 sola capa interna y 1 sola neurona
# funcion de activacion logistica

library(neuralnet)
mod1 <- neuralnet(y ~ x1 + x2, data=datos,
                  hidden=c(3, 2),
                  rep=3,
                  linear.output = FALSE)

# Dibujando la red entrenada
plot(mod1, rep = 1)
plot(mod1, rep = 2)
plot(mod1, rep = 3)
plot(mod1, rep = 'best')

# Para conocer la clase del objeto mod1
class(mod1)

# Para ver los objetos dentro de mod1
names(mod1)

# Explorando los pesos para luego hacer operaciones con ellos
mod1$weights

# Haciendo las predicciones automaticamente
probab <- predict(mod1, newdata=datos)
colnames(probab) <- levels(datos$y)
head(probab)

# Para ubicar la columna en la cual esta la maxima probabilidad
i_max_prob <- apply(X=probab, MARGIN=1, FUN=which.max)

y_hat <- levels(datos$y)[i_max_prob]

# Matriz de confusion
table(Predict=y_hat, True=datos$y)

# Identificando las observaciones mal clasificadas
with(datos, plot(x=x1, y=x2, col=c("tomato", "dodgerblue")[y], 
                 pch=20, las=1))

error <- datos$y != y_hat
points(x=datos$x1[error], y=datos$x2[error], lwd=2)

