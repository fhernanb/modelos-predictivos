# -------------------------------------------------------------------------
# En este ejemplo se busca encontrar un modelo de regresion que explique 
# la variable respuesta y en función de las covariables x1 a x11, los datos 
# provienen del ejercicio 9.5 del libro de Montgomery, Peck and Vining (2003).
# El paquete MPV (Braun 2019) contiene todos los datos que acompañan al libro.
# -------------------------------------------------------------------------

# Los datos a usar estan disponibles en un repositorio de github
file <- "https://raw.githubusercontent.com/fhernanb/datos/master/table_b3_MASS.txt"
datos <- read.table(file, header=TRUE)
head(datos)

# Exploremos las filas 23 y 25 porque hay NA
datos[c(23, 25), ]

# Vamos a eliminar las filas que tienen nNA
datos <- table.b3[-c(23, 25), ]

library(MPV)  # Aqui estan los datos
table.b3[22:26, ] # Can you see the missing values?

datos <- table.b3[-c(23, 25), ]

# Vamos a construir el modelo
library(rpart)
library(rpart.plot)

mod1 <- rpart(y ~ ., data=datos)

# Para dibujar el arbol
prp(mod1)

# Vamos a estimar y
y_hat <- predict(object=mod1, newdata=datos)

# Vamos a calcular cor y mse entre y_hat y el verdadero y
cor(y_hat, datos$y)
mean((datos$y - y_hat)^2)


