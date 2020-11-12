# -------------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) para mostrar
# el efecto que tienen los hiper-parametros de un kernel polinomial
# en una svm para regresion
# -------------------------------------------------------------------------


# Creando los datos -------------------------------------------------------
set.seed(1234)
x <- sort(runif(n=40, min=0, max=5)) # sort for convenience
set.seed(1234)
y <- sin(x) + rnorm(40, sd=0.3)

# Construyendo el dataframe
datos <- data.frame(x=x, y=y)

# Diagrama de dispersion
plot(x, y, pch=20, las=1)



# Creando la animacion ----------------------------------------------------
library(kernlab)

# A continuacion los vectores con los valores de los hiper-parametros
# que vamos asar en la animacion
my_degree <- 1:10
my_scale <- seq(from=-1.5, to=1.5, by=0.1)
my_offset <- seq(from=-1.5, to=1.5, by=0.1)

# La siguiente es la funcion para crear el modelo, obtener y_hat
# y hacer el dibujo. Los valores por defecto sera 1.
my_fun <- function(p1=1, p2=1, p3=1, pausa=0.5) {
  mod_pol <- ksvm(y ~ x, data=datos,
                  type="eps-svr", kernel="polydot",
                  C=1, epsilon=0.1,
                  kpar=list(degree=p1, scale=p2, offset=p3))
  y_hat_pol <- predict(mod_pol)
  titulo <- paste0("Degree=", p1, ", scale=", p2, " y offset=", p3)
  plot(x, y, pch=20, las=1, main=titulo)
  points(x=x, y=y_hat_pol, type="l", lwd=4, col="deepskyblue3")
  Sys.sleep(time=pausa) # Para pausar entre graficos (segundos)
}

# Para explorar efecto del grado
for (i in my_degree) my_fun(p1=i)

# Para explorar efecto del scale
for (i in my_scale) my_fun(p2=i)

# Para explorar efecto del offset
for (i in my_offset) my_fun(p3=i)

# Para explorar el efecto de todo
for (i in my_degree)
  for (j in my_scale)
    for (k in my_offset) 
      my_fun(p1=i, p2=j, p3=k, pausa=0.05)
