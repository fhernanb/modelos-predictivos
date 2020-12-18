# -------------------------------------------------------------------------
# En este ejemplo se busca encontrar un modelo de regresion que explique 
# la variable respuesta y en función de las covariables x1 a x11, los datos 
# provienen del ejercicio 9.5 del libro de Montgomery, Peck and Vining (2003).
# El paquete MPV (Braun 2019) contiene todos los datos que acompañan al libro.
# -------------------------------------------------------------------------

# Los datos a usar estan disponibles en un repositorio de github
file <- "https://raw.githubusercontent.com/fhernanb/datos/master/table_b3_MPV.txt"
datos <- read.table(file, header=TRUE)
head(datos)

# Exploremos las filas 23 y 25 porque hay NA
datos[c(23, 25), ]

# Vamos a eliminar las filas que tienen nNA
datos <- datos[-c(23, 25), ]

# Para ver la dimension de los datos
dim(datos)

# Vamos a construir el modelo
library(gbm)
mod <- gbm(y ~ x1 + x2, data=datos,
           n.trees=180,
           n.minobsinnode=3)

# Estimando y usando los datos de entrenamiento
y_hat <- predict(mod, datos)

# Para ver algunas medidas de desempeño
cor(datos$y, y_hat)

MLmetrics::R2_Score(y_pred=y_hat, y_true=datos$y)
MLmetrics::MSE(y_pred=y_hat, y_true=datos$y)

# Agregando el modelo estimado al diagrama originar
plot(x=datos$y, y=y_hat, las=1)

