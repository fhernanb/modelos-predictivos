# -------------------------------------------------------------------------
# En este script se muestra como crear un kernel polinomial manual
# para un problema de clasificacion.
# y el efecto de los kernel
# -------------------------------------------------------------------------

# A continuacion los datos que vamos a usar

x1 <- c(1,1,2,3,3,6,6,6,9,9,10,11,12,13,16,18)
x2 <- c(18,13,9,6,15,11,6,3,5,2,10,5,6,1,3,1)
datos <- data.frame(x1=x1, x2=x2)

# La variable respuesta (y o grupo) se muestra a continuacion
y <- c(-1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1)
grupo <- c("red", "red", "red", "red", "green4", "green4",
           "green4", "red", "green4", "red", "green4", 
           "green4", "green4", "red", "green4", "red")

# Dibujemos los datos
with(datos, plot(x=x1, y=x2, las=1, pch=20, col=grupo))

# Aqui vamos a usar un kernel que convierte la pareja
# (x1, x2) a la terna (x1^2, sqrt(2) * x1 * x2, x2^2)

my_kernel <- function(datos) {
  a <- datos[, 1]
  b <- datos[, 2]
  z1 <- a^2
  z2 <- sqrt(2) * a * b
  z3 <- b^2
  data.frame(z1, z2, z3)
}

# Los datos transformados
datos_transf <- my_kernel(datos)
datos_transf

# Vamos a crear un diagrama dispersion 3d, muevalo con el mouse!!!
library(rgl)
with(datos_transf, plot3d(x=z1, y=z2, z=z3, col=grupo, size=10))

# Primera forma
# Vamos autilizar los datos transformados para obtener 
# la matriz cuadrada simetrica de transformacion
x_transf <- as.matrix(datos_transf)
x_transf %*% t(x_transf)

# Segunda forma para encontrar la matriz
library(kernlab)
my_poli <- polydot(degree=2, scale=1, offset=0)

my_poli(c(1, 18), c(1, 18))
my_poli(c(1, 18), c(1, 13))
my_poli(c(1, 18), c(2, 9))
# ... y asi sucesivamente

# Tercera forma de encontrar la matriz
kernelMatrix(kernel=my_poli, x=as.matrix(datos))

# Aqui se muestra como incluir kernel manualmente -------------------------

x1 <- c(1,1,2,3,3,6,6,6,9,9,10,11,12,13,16,18)
x2 <- c(18,13,9,6,15,11,6,3,5,2,10,5,6,1,3,1)
x <- cbind(x1, x2)
y <- c(-1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1)

# Para ver nuevamente los datos
cbind(x, y)

# Vamos a convertir a la matriz x en x_transf usando polydot
# y luego vamos a crear una svm "lineal"
my_poli <- polydot(degree=2, scale=1, offset=0)
x_transf <- kernelMatrix(kernel=my_poli, x=x)

fit1 <- ksvm(x=x_transf, y=y, kernel="vanilladot",
             scaled=FALSE,
             type="C-svc")

y_hat1 <- predict(fit1, type="response")

# Vamos a pedirle a ksvm que use polydot
fit2 <- ksvm(x=x, y=y, kernel="polydot",
             scaled=FALSE,
             kpar=list(degree=2, scale=1, offset=0),
             type="C-svc")

y_hat2 <- predict(fit2, type="response")

# NOTA: en fit1 se uso x_transf mientras que en fit2 se uso x. 
# diferencia esta en el kernel.

# Comparemos las estimaciones para verificar que coinciden
cbind(y_hat1, y_hat2)

# Construyamos una tabla
table(auto=y_hat1, manual=y_hat2)

# Construyamos una tabla de CONFUSION (usando y_hat1 o y_hat2)
table(real=y, estimado=y_hat1)

# Que sucede si usamos kernel="vanilladot" y no transformamos?
fit3 <- ksvm(x=x, y=y, kernel="vanilladot",
             scaled=FALSE,
             type="C-svc")

y_hat3 <- predict(fit3, type="response")

# Comparemos las estimaciones
cbind(y_hat1, y_hat2, y_hat3)

# Comparemos con los valores reales
table(real=y, estimado=y_hat3)


# Explorando los kernel ---------------------------------------------------
library(kernlab)

x1 <- c(1,1,2,3,3,6,6,6,9,9,10,11,12,13,16,18)
x2 <- c(18,13,9,6,15,11,6,3,5,2,10,5,6,1,3,1)
x <- cbind(x1, x2)
y <- c(-1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1)

# kernel lineal
mod <- ksvm(x, y, type="C-svc", kernel="vanilladot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# kernel radial
mod <- ksvm(x, y, type="C-svc", kernel="rbfdot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# kernel polinomial
mod <- ksvm(x, y, type="C-svc", kernel="polydot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# kernel tanhdot
mod <- ksvm(x, y, type="C-svc", kernel="tanhdot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# kernel laplacedot
mod <- ksvm(x, y, type="C-svc", kernel="laplacedot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# kernel besseldot
mod <- ksvm(x, y, type="C-svc", kernel="besseldot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# kernel anovadot
mod <- ksvm(x, y, type="C-svc", kernel="anovadot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# kernel splinedot
mod <- ksvm(x, y, type="C-svc", kernel="splinedot")
y_hat <- predict(mod, type="response")
table(y_hat, y)
plot(mod, data=x)

# Tarea:
# Revisar este enlace 
# https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel
