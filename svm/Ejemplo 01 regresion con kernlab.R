# -------------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) para mostrar
# el uso de svm en regresion
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

# Funcion para calcular MSE -----------------------------------------------
mse <- function(y, y_hat) mean((y - y_hat)^2)

# svm lineal --------------------------------------------------------------
library(kernlab)

# Para ajustar el modelo
mod_lin <- ksvm(y ~ x, type="eps-svr", kernel="vanilladot",
                C=1, epsilon=0.1)

# To obtain y_hat
y_hat_lin <- predict(mod_lin)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_lin)
mse(y, y_hat_lin)

# To illustrate the results
plot(x, y, pch=20)
points(x=x, y=y_hat_lin, type="l", lwd=6, col="red")


# svm polinomial ----------------------------------------------------------

# Para ajustar el modelo con los hiper-parametros por defecto
mod_pol <- ksvm(y ~ x, type="eps-svr", kernel="polydot",
                C=1, epsilon=0.1, 
                kpar=list(degree=1, scale=1, offset=1))

# To obtain y_hat
y_hat_pol <- predict(mod_pol)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_pol)
mse(y, y_hat_pol)

# To illustrate the results
plot(x, y, pch=20)
points(x=x, y=y_hat_pol, type="l", lwd=2, col="blue")


# svm radial --------------------------------------------------------------

# Para ajustar el modelo con los hiper-parametros por defecto
mod_rad <- ksvm(y ~ x, type="eps-svr", kernel="rbfdot",
                C=1, epsilon=0.1,
                kpar=list(sigma=1))

# To obtain y_hat
y_hat_rad <- predict(mod_rad)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_rad)
mse(y, y_hat_rad)

# To illustrate the results
plot(x, y, pch=20)
points(x=x, y=y_hat_rad, type="l", lwd=2, col="forestgreen")

# Comparing ---------------------------------------------------------------
plot(x, y, pch=20, las=1)
points(x=x, y=y_hat_lin, type="l", lwd=6, col="red")
points(x=x, y=y_hat_pol, type="l", lwd=2, col="blue")
points(x=x, y=y_hat_rad, type="l", lwd=2, col="forestgreen")
legend("topright", lty=1,
       col=c("red", "blue", "forestgreen"),
       legend=c("Linear", "Polynomial", "Radial"))
