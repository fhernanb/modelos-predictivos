# -------------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) para mostrar
# el entrenamiento de una svm para regresion con caret
# -------------------------------------------------------------------------


# Creando los datos -------------------------------------------------------
set.seed(1234)
x <- sort(runif(n=40, min=0, max=5)) # sort for convenience
set.seed(1234)
y <- sin(x) + rnorm(40, sd=0.3)
plot(x, y, pch=20, las=1)

# Creando el dataframe con los datos
datos <- data.frame(y, x)


# Tuning svm lineal -------------------------------------------------------
library(caret)

# Control parameters for train
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)

# Using my own grid -------------------------------------------------------
# Aqui vamos a elegir nosotros mismo los valores de los
# hiper parametros para buscar la combinacion que optimize la metrica

my_grid <- expand.grid(C=c(0.1, 0.5, 1, 1.5))

# To control the re-sampling
set.seed(825) 

# To train the model
fit1 <- train(y ~ x, 
              data = datos, 
              method = "svmLinear", 
              metric = "RMSE",
              trControl = fitControl,
              tuneGrid = my_grid)

# To show the results
fit1

# To plot the results
plot(fit1)

# To explore the best model
fit1$bestTune


# usando un grid mas fino -------------------------------------------------

my_grid <- expand.grid(C=seq(from=0.1, to=1, by=0.05))

set.seed(825)
fit2 <- train(y ~ x, 
              data = datos, 
              method = "svmLinear", 
              metric = "RMSE",
              trControl = fitControl,
              tuneGrid = my_grid)

# To show the results
fit2

# To plot the results
plot(fit2)

# To explore the best model
fit2$bestTune

