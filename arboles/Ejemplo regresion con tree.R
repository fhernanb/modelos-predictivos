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
library(tree)
mod_tree <- tree(y ~ ., data=datos)

# Para ver un resumen
summary(mod_tree)

# Para dibujar el arbol
plot(mod_tree)
text(mod_tree, pretty=0)

# Vamos a estimar y
y_hat <- predict(object=mod_tree, newdata=datos)

# Vamos a calcular cor y mse entre y_hat y el verdadero y
cor(y_hat, datos$y)
mean((datos$y - y_hat)^2)

# Vamos a dibujar las predicciones versus los valores ajustados
plot(x=datos$y, y=y_hat, pch=20, las=1)
abline(a=0, b=1, col='tomato', lty='dashed')

# Usando 
cv_mod <- cv.tree(mod_tree)
plot(cv_mod)

# Explorando los argumentos internos --------------------------------------

# La funcion tree tiene el argumento control que se puede modificar
# para personalizar el arbol. Consultar la ayuda de tree.control.

# Abajo la estructura de tree.control

# tree.control(nobs, mincut = 5, minsize = 10, mindev = 0.01)

# Vamos a crear un arbol con maxima profundidad de 5
# y 3 como minimo numero de observaciones para dividir nodo.
mod_tree_custom <- tree(y ~ ., data=datos,
                        control=tree.control(nobs=nrow(datos),
                                             mincut=0.5,
                                             minsize=3))

plot(mod_tree_custom)
text(mod_tree_custom, all=TRUE, cex=0.7)
