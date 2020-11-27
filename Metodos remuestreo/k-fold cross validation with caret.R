# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar a aplicar k-fold cross validation 
# usando la base de datos Auto de ISLR.
# Lo vamos a realizar de forma manual y automatica
# y vamos a crear un lm para explicar mpg en funcion de 
# horsepower y de horsepower^2
# Metrica a usar: rmse
# -------------------------------------------------------------------------


# Los datos que vamos a usar
library(ISLR)
head(Auto)

# Vamos a explorar los datos
library(tidyverse)
Auto %>% glimpse()


# Manualmente -------------------------------------------------------------

# Vamos a crear un vector para identificar los folds o particiones
folds <- rep(1:10, each=39)

# Vamos a usar solo las obs 1 a 390, las ultimas dos NO!!!
datos <- Auto[1:390, ]

# Vectro vacio para almacernar los rmse
rmse <- numeric(10)

# Vamos a recorrer los folds y calcular la medida
for (i in 1:10) {
  testIndexes <- which(folds == i, arr.ind=TRUE)
  testData  <- datos[ testIndexes, ]
  trainData <- datos[-testIndexes, ]
  mod <- glm(mpg ~ poly(horsepower, degree=2), data=trainData)
  y_hat <- predict(object=mod, newdata=testData)
  rmse[i] <- sqrt(mean((testData$mpg - y_hat)^2))
}

# Para ver los rmse
rmse

# Para ver la distribucion de los rmse
plot(density(rmse))
rug(rmse, col='tomato')

# Para ver la media de los rmse
rmse %>% mean()

# Para ver la varianza de los rmse
rmse %>% var()


# Automaticamente teniendo control de los fold ----------------------------

library(caret)

# Matriz con los i de las observaciones
x <- matrix(1:390, ncol=10)

# Creando una lista con los folds
index_to_test <- split(x=x, f=rep(1:ncol(x), each=nrow(x)))
index_to_train <- lapply(index_to_test, function(x) setdiff(1:390, x))

# Vamos a chequear lo que hay dentro de los objetos
index_to_test
index_to_train

# Definiendo
fitControl <- trainControl(method = "cv",
                           savePredictions=TRUE,
                           index = index_to_train,
                           indexOut = index_to_test)

# To train the model
fit1 <- train(mpg ~ poly(horsepower, degree=2), 
              data = datos, 
              method = "glm", 
              metric = "RMSE",
              trControl = fitControl)

# To show the results
fit1

# Comparemos con el resultado manual
mean(rmse)

# Para ver los resultados para cada fold
fit1$resample

# Comparemos con el resultado manual
rmse

# Para extraer los rmse individuales
fit1$resample$RMSE

# Para ver las predicciones, aqui pred=y_hat obs=y_true
pred <- fit1$pred
pred$pred[1:5]



# Automaticamente con k=10 ------------------------------------------------

library(caret)

k <- 10
fitControl <- trainControl(method = "cv",
                           number = k)

# To train the model
fit2 <- train(mpg ~ poly(horsepower, degree=2), 
              data = datos, 
              method = "glm", 
              metric = "RMSE",
              trControl = fitControl)

# To show the results
fit2

# Para ver los resultados para cada fold
fit2$resample

# Para ver la media
fit2$resample$RMSE %>% mean()

# Para ver la varianza
fit2$resample$RMSE %>% var()

# Para ver la distribucion de los rmse
plot(density(fit2$resample$RMSE), main='Densidad', las=1)
rug(fit2$resample$RMSE, col='tomato')

