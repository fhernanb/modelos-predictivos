# -------------------------------------------------------------------------
# En este script se exploran algunas funciones que nos ayuda
# a encontrar los vecinos mas cercanos
# -------------------------------------------------------------------------

# Vamos a crear unos datos ficticios
datos <- data.frame(x1=c(3, 4, 1, 5, 2), 
                    x2=c(2, 7, 7, 4, 3))

row.names(datos) <- LETTERS[1:5]

# Dibujando los datos
plot(datos, las=1, pch=row.names(datos), ylim=c(0, 8))
grid()

# Calculando las distancias euclideanas pero se puede usar otra
d <- dist(datos, method = "euclidean", diag=TRUE, upper=TRUE)
print(d, digits=2)

# Usando FNN --------------------------------------------------------------
library(FNN)
get.knn(datos, k=3)


# Creando mi propia funcion para ingresar la matriz de distancias ---------

# Vamos a crear una funcion sencilla que recibe la matriz
# de distancias y el numero k de vecinos
veci_mas_cerc <- function(d, k) {
  d <- as.matrix(d)
  diag(d) <- NA # Modificando la diagonal
  veci <- apply(d, 2, order) # aplicando order
  veci <- t(veci)
  return(veci[, 1:k])
}

veci_mas_cerc(d, k=3)


