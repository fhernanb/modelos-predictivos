# -------------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) para mostrar
# el uso de svm en regresion
# -------------------------------------------------------------------------


# Creando los datos -------------------------------------------------------
set.seed(1234)
x <- sort(runif(n=40, min=0, max=5)) # sort for convenience
set.seed(1234)
y <- sin(x) + rnorm(40, sd=0.3)
plot(x, y, pch=20, las=1)


# Funcion para calcular MSE -----------------------------------------------
mse <- function(y, y_hat) mean((y - y_hat)^2)

# svm lineal --------------------------------------------------------------
library(e1071)

# Para ajustar el modelo
mod_lin <- svm(y ~ x, type="eps-regression", kernel="linear",
               cost=1, epsilon=0.1)

# To obtain y_hat
y_hat_lin <- predict(mod_lin)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_lin)
mse(y, y_hat_lin)

# To illustrate the results
plot(x, y, pch=20)
points(x=x, y=y_hat_lin, type="l", lwd=2, col="red")


# svm polinomial ----------------------------------------------------------

# Para ajustar el modelo
mod_pol <- svm(y ~ x, type="eps-regression", kernel="polynomial",
               cost=1, epsilon=0.1)

# To obtain y_hat
y_hat_pol <- predict(mod_pol)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_pol)
mse(y, y_hat_pol)

# To illustrate the results
plot(x, y, pch=20)
points(x=x, y=y_hat_pol, type="l", lwd=2, col="blue")


# svm radial --------------------------------------------------------------

# Para ajustar el modelo
mod_rad <- svm(y ~ x, type="eps-regression", kernel="radial",
               cost=1, epsilon=0.1)

# To obtain y_hat
y_hat_rad <- predict(mod_rad)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_rad)
mse(y, y_hat_rad)

# To illustrate the results
plot(x, y, pch=20)
points(x=x, y=y_hat_rad, type="l", lwd=2, col="forestgreen")

# Comparing ---------------------------------------------------------------
plot(x, y, pch=20)
points(x=x, y=y_hat_lin, type="l", lwd=2, col="red")
points(x=x, y=y_hat_pol, type="l", lwd=2, col="blue")
points(x=x, y=y_hat_rad, type="l", lwd=2, col="forestgreen")
legend("topright", lty=1,
       col=c("red", "blue", "forestgreen"),
       legend=c("Linear", "Polynomial", "Radial"))


# Tuning parameters -------------------------------------------------------

# Vamos a sintonizar los parametros de svm lineal
lin_tune <- tune.svm(y~x, kernel="linear",
                    cost=c(0.1, 0.5, 1, 1.5),
                    epsilon=c(0.1, 0.5, 1, 1.5))
summary(lin_tune)

# Vamos a sintonizar los parametros de svm polinomial
pol_tune <- tune.svm(y~x, kernel="polynomial",
                     degree=c(2, 3, 4),
                     gamma=c(0.1, 1, 2),
                     coef0=c(0.1, 0.5, 1, 2, 3),
                     cost=c(0.1, 0.5, 1, 1.5),
                     epsilon=c(0.1, 0.5, 1, 1.5))
summary(pol_tune)

# Vamos a sintonizar los parametros de svm radial
rad_tune <- tune.svm(y~x, kernel="radial",
                     gamma=c(0.1, 0.5, 1, 1.5, 2),
                     cost=c(0.1, 0.5, 1, 1.5),
                     epsilon=c(0.1, 0.5, 1, 1.5))
summary(rad_tune)

# Identificando los valores de los hipeparametros que mejoran 
# cada modelo

lin_tune$best.model
pol_tune$best.model
rad_tune$best.model

# Los mejores modelos -----------------------------------------------------

# El mejor lineal
best_lin <- lin_tune$best.model
y1 <- predict(best_lin)
cor(y, y1)
mse(y, y1)

# El mejor polinomial
best_pol <- pol_tune$best.model
y2 <- predict(best_pol)
cor(y, y2)
mse(y, y2)

# El mejor radial
best_rad <- rad_tune$best.model
y3 <- predict(best_rad)
cor(y, y3)
mse(y, y3)


# Comparando sin tuning y con tuning --------------------------------------

par(mfrow=c(1, 2))

plot(x, y, pch=20, main="Default", las=1)
points(x=x, y=y_hat_lin, type="l", col="red")
points(x=x, y=y_hat_pol, type="l", col="blue")
points(x=x, y=y_hat_rad, type="l", col="forestgreen")
legend("topright", lty=1, bty="n",
       col=c("red", "blue", "forestgreen"),
       legend=c("Linear", "Polynomial", "Radial"))

plot(x, y, pch=20, main="Best tune parameters", las=1)
points(x=x, y=y1, type="l", col="red")
points(x=x, y=y2, type="l", col="blue")
points(x=x, y=y3, type="l", col="forestgreen")

