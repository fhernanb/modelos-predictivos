# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar la base de datos Cars93
# del paquete MASS para estimar el precio del auto en funcion del
# peso y del rendimiento del combustible
# -------------------------------------------------------------------------

# Los datos que vamos a usar
library(MASS)

# Vamos a explorar los datos
library(tidyverse)
Cars93 %>% glimpse()

# Diagrama de dispersion
library(plotly)
Cars93 %>% plot_ly(x=~Weight, y=~MPG.city, z=~Price, color=~Price)

# Vamos a seleccionar las variables de interes
# de los primeros 10 autos
Cars93 %>% select(Weight, MPG.city) %>% slice(1:10) -> x_train
Cars93 %>% select(Price) %>% slice(1:10) %>% pull() -> y_train

# Vamos a dibujar las valores de las variables
plot(x_train, las=1, pch=20)

# Supongamos que tenemos una nueva observacion para la 
# cual queremos estimar el precio
new_x <- data.frame(Weight=3500, MPG.city=21)

# Dibujemos los datos de entrenamiento y la nueva observacion
plot(x_train, pch=20, las=1)
grid()
points(new_x, col="red", pch=20)
points(x=3496, y=19, pch=1, cex=2, col="blue")

# Vamos a aplicar k=1 NN a ojo
k <- 1
identify(x_train, n=k, labels=y_train)

# Vamos a aplicar k=1 NN usando distancias

# Primero juntemos las base de datos
library(dplyr)
new_df <- bind_rows(x_train, new_x)

# Calculemos las distancias
d <- dist(new_df)
print(d, digits=0)

# Para obtener y_hat
y_train[9]

# Usando el paquete FNN ---------------------------------------------------
library(FNN)

fit <- knn.reg(train=x_train, y=y_train, test=new_x, k=1)
fit$pred
