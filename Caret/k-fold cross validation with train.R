# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar a aplicar k-fold cross validation 
# usando la base de datos Auto de ISLR.
# Lo vamos a realizar de forma manual y automatica
# -------------------------------------------------------------------------


# Los datos que vamos a usar
library(ISLR)
head(Auto)

# Vamos a explorar los datos
library(tidyverse)
Auto %>% glimpse()


# Manualmente -------------------------------------------------------------

folds <- rep(1:10, each=39)

datos <- Auto[1:390, ]

mse <- numeric(10)

for (i in 1:10) {
  testIndexes <- which(folds == i, arr.ind=TRUE)
  testData  <- datos[ testIndexes, ]
  trainData <- datos[-testIndexes, ]
  mod <- glm(mpg ~ poly(horsepower, degree=2), data=trainData)
  y_hat <- predict(object=mod, newdata=testData)
  mse[i] <- mean((testData$mpg - y_hat)^2)
}

mse
mean(mse)


# Tunning the model -------------------------------------------------------
library(caret)

# Control parameters for train
x <- matrix(1:390, ncol=10)
my_index <- split(x, rep(1:ncol(x), each=nrow(x))) # lista con los folds

fitControl <- trainControl(method = "cv",
                           savePredictions=TRUE,
                           indexOut = my_index)

# To train the model
fit1 <- train(mpg ~ poly(horsepower, degree=2), 
              data = datos, 
              method = "lm", 
              metric = "RMSE",
              trControl = fitControl)

# To show the results
fit1

# Para ver los resultados para cada fold
fit1$resample

# Para extraer los rmse individuales
fit1$resample$RMSE
mean(fit1$resample$RMSE)

# Para extraer los mse individuales
fit1$resample$RMSE^2
mean(fit1$resample$RMSE^2)

# Para ver las predicciones, aqui pred=y_hat obs=y_true
pred <- fit1$pred
pred$pred[1:5]


# usando k=10 -------------------------------------------------------------
library(caret)

fitControl <- trainControl(method = "cv",
                           number = 10)

# To train the model
fit2 <- train(mpg ~ poly(horsepower, degree=2), 
              data = datos, 
              method = "lm", 
              metric = "RMSE",
              trControl = fitControl)

# To show the results
fit2

# Para ver los resultados para cada fold
fit2$resample

