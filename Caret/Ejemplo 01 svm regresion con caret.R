# -------------------------------------------------------------------
# En este ejemplo se usan datos artificiales (simulados) para mostrar
# el entrenamiento de una svm con caret
# -------------------------------------------------------------------


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

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10)

my_svm_lin_Grid <- expand.grid(C=c(0.1, 0.5, 1, 1.5))

set.seed(825)
svm_lin <- train(y ~ x, data = datos, 
                 method = "svmLinear", 
                 metric = "RMSE",
                 trControl = fitControl,
                 tuneGrid = my_svm_lin_Grid)
svm_lin
plot(svm_lin)

# usando un grid mas fino
my_svm_lin_Grid <- expand.grid(C=seq(from=0.1, to=1, by=0.05))

set.seed(825)
svm_lin <- train(y ~ x, data = datos, 
                 method = "svmLinear", 
                 metric = "RMSE",
                 trControl = fitControl,
                 tuneGrid = my_svm_lin_Grid)
svm_lin
plot(svm_lin)

