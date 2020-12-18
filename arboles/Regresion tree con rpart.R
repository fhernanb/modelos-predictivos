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
library(rpart)
mod_rpart <- rpart(y ~ ., data=datos)

# Para dibujar el arbol con la funcion prp de rpart.plot
library(rpart.plot)
prp(mod_rpart)

# Para dibujar el arbol con la funcion fancyRpartPlot de rattle
library(rattle)
fancyRpartPlot(mod_rpart, palettes='Reds')

# Vamos a estimar y
y_hat <- predict(object=mod_rpart, newdata=datos)

# Vamos a calcular cor y mse entre y_hat y el verdadero y
cor(y_hat, datos$y)
mean((datos$y - y_hat)^2)

# Vamos a dibujar las predicciones versus los valores ajustados
plot(x=datos$y, y=y_hat, pch=20, las=1)
abline(a=0, b=1, col='tomato', lty='dashed')


# Explorando los argumentos internos --------------------------------------

# La funcion rpart tiene el argumento control que se puede modificar
# para personalizar el arbol. Consultar la ayuda de rpart.control.

# Abajo la estructura de rpart.control

# rpart.control(minsplit = 20, minbucket = round(minsplit/3), 
#               cp = 0.01, maxcompete = 4, maxsurrogate = 5, 
#               usesurrogate = 2, xval = 10,
#               surrogatestyle = 0, maxdepth = 30, ...)

# Vamos a crear un arbol con maxima profundidad de 5
# y 3 como minimo numero de observaciones para dividir nodo.
mod_rpart_custom <- rpart(y ~ ., data=datos,
                          control=rpart.control(maxdepth=4,
                                                minsplit=3))

fancyRpartPlot(mod_rpart_custom, palettes='RdPu')

