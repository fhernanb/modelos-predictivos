# -------------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) mostrando
# el uso de caret para sintonizar los hiper-parametros
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


# svm polinomial ----------------------------------------------------------
library(kernlab)

# Para ajustar el modelo con los hiper-parametros por defecto
mod_pol <- ksvm(y ~ x, data=datos, type="eps-svr", kernel="polydot",
                C=1, epsilon=0.1, 
                kpar=list(degree=1, scale=1, offset=1))

# To obtain y_hat
y_hat_pol <- predict(mod_pol)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_pol)
mse(y, y_hat_pol)

# To illustrate the results
plot(x, y, pch=20, las=1)
points(x=x, y=y_hat_pol, type="l", lwd=2, col="blue")

# Usando caret ------------------------------------------------------------
library(caret)

# Using my own grid -------------------------------------------------------
# Aqui vamos a elegir nosotros mismos los valores de los
# hiper-parametros para buscar la combinacion que optimize la metrica

# Control parameters for train
fitControl <- trainControl(method = "cv",
                           number = 7)

my_grid <- expand.grid(degree=c(1, 2, 3),
                       scale=c(0.5, 1, 1.5),
                       C=c(1, 3, 5, 7))

# To control the re-sampling
set.seed(825)

# To train the model
fit1 <- train(y ~ x, 
              data = datos, 
              method = "svmPoly", 
              metric = "RMSE",
              trControl = fitControl,
              tuneGrid = my_grid)

# To show the results
fit1

# To plot the results
plot(fit1)

# To explore the best model
fit1$bestTune

# To obtain y_hat
y_hat_pol_tuned <- predict(fit1)

# To obtain the correlation coefficient and MSE.
cor(y, y_hat_pol_tuned)
mse(y, y_hat_pol_tuned)

# To illustrate the results
plot(x, y, pch=20, las=1)
points(x=x, y=y_hat_pol_tuned, type="l", lwd=2, col="red")

# To compare the two models
plot(x, y, pch=20, las=1)
points(x=x, y=y_hat_pol, type="l", lwd=2, col="blue")
points(x=x, y=y_hat_pol_tuned, type="l", lwd=2, col="red")

# Comparando los dos modelos
mod_pol
fit1$finalModel

# Moraleja: usar kernel sin sintonizar los hiper-parametros no es bueno.
