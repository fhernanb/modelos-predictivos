# -------------------------------------------------------------------------
# En este script se ilustra el ejemplo del video
# https://youtu.be/Toet3EiSFcM
# -------------------------------------------------------------------------

# Datos originales parecidos a los mostrados en el video
x <- c(0.4, 0.8, 1.2, 1.9, 2.4, 2.5, 2.8, 
       2.9, 3.1, 3.2, 3.3, 4.1, 4.4, 5.3)

col <- c(rep("red", 4), rep("green4", 7), rep("red", 3))

plot(x=x, y=rep(0, 14), las=1, xlab="x", ylab="",
     ylim=c(0, 2), pch=20, cex=2, col=col)

# Vamos ahora a crear una nueva variable y
y <- x^2

plot(x=x, y=y, las=1, xlab="x", ylab="y",
     pch=20, cex=2, col=col)

# Vamos a usar un kernel polinomial
my_kernel_poly <- function(a, b, r, d) (a * b + r)^d

# Hagamos una prueba con
r <- 0.5
d <- 2

new_y <- my_kernel_poly(a=x, b=x, r=r, d=d)

plot(x=x, y=new_y, las=1, xlab="x", ylab="y",
     pch=20, cex=2, col=col)

# Hagamos otra prueba con
r <- 1
d <- 2

new_y <- my_kernel_poly(a=x, b=x, r=r, d=d)

plot(x=x, y=new_y, las=1, xlab="x", ylab="new_y",
     pch=20, cex=2, col=col)


# Vamos a usar el paquete kernlab para calcular los valores
library(kernlab)
my_poli <- polydot(degree=2, scale=1, offset=0.5)
my_poli

# Para obtener el valor 16002.25 mostrado en el minuto 6:46 del video
my_poli(9, 14)
