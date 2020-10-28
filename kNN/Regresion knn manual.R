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

# Como conjunto de entrenamiento vamos a usar las primeras 10 observaciones.
Cars93 %>% select(Weight, MPG.city) %>% slice(1:10) -> x_train
Cars93 %>% select(Price) %>% slice(1:10) %>% pull() -> y_train

# Vamos a dibujar un diagrama de dispersion
plot(x_train, las=1, pch=20)

# Una modificacion del diagrama anterior
plot(x_train, las=1, pch=20, 
     main="El valor de y se \n representa por la tonalidad",
     col=gray(y_train/max(y_train)))

# Supongamos que tenemos una nueva observacion para la 
# cual queremos estimar el precio, la nueva observacion es:
new_x <- data.frame(Weight=3500, MPG.city=21)

# Dibujemos los datos de entrenamiento y la nueva observacion
plot(x_train, pch=20, las=1)
grid()
points(new_x, col="red", pch=20)
points(x=3496, y=19, pch=1, cex=2, col="blue")
text(x=3550, y=19.5, adj=0, label="Veci eucli cercano", col="blue")

# Vamos a identificar el vecino mas cercano a ojo
k <- 1
identify(x_train, n=k, labels=y_train)

# Vamos a encontrar las distancias entre las obs de train y new_x.
# Primero juntemos las base de datos x_train y new_x en una sola
# para calcular las distancias entre new_x y las observaciones de x_train
library(dplyr)
new_df <- bind_rows(x_train, new_x) # Nuestra new_x estara en j=11

# Calculemos las distancias con la funcion dist, consulte la ayuda
# de la funcion dist() para conocer otros detalles.
d <- dist(new_df, method="euclidean")
print(d, digits=0)
# de la salida anterior vemos que las obs mas cercanas son 9 y 7.

# Vamos a aplicar knn con k=1 usando distancia euclideana
# Para obtener y_hat
y_train[9]

# Vamos a aplicar knn con k=2 usando distancia euclideana
# Para obtener y_hat
(y_train[9] + y_train[7]) / 2

# Usando el paquete FNN ---------------------------------------------------
library(FNN)

# En esta parte del ejemplo vamos a aplicar knn usando un paquete
# y luego vamos a comparar con los resultados de forma manual.

# Vamos a ajustar el modelo y luego a predecir usando k=1
fit1 <- knn.reg(train=x_train, y=y_train, test=new_x, k=1)
fit1$pred

# Vamos a ajustar el modelo y luego a predecir usando k=2
fit2 <- knn.reg(train=x_train, y=y_train, test=new_x, k=2)
fit2$pred



