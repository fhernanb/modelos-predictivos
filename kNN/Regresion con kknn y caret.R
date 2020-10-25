# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar la base de datos Cars93
# del paquete MASS para estimar el precio del auto en funcion del
# peso y del rendimiento del combustible
# -------------------------------------------------------------------------


# Los datos que vamos a usar
library(MASS)
head(Cars93)

# Vamos a explorar los datos
library(tidyverse)
Cars93 %>% glimpse()

# Diagrama de dispersion
library(plotly)
Cars93 %>% plot_ly(x=~Weight, y=~MPG.city, z=~Price, color=~Price)

# Particion de los datos, vamos a usar aprox 60% y 40% para train y test
i_train <- sample(1:93, size=60)

train_data <- Cars93[i_train, ] # 60 obs
test_data <- Cars93[-i_train, ] # 33 obs

# Tunning the model -------------------------------------------------------
library(caret)

# Control parameters for train
fitControl <- trainControl(method = "cv",
                           number = 7)

# Using my own grid -------------------------------------------------------
# Aqui vamos a elegir nosotros mismos los valores de los
# hiper-parametros para buscar la combinacion que optimize la metrica

my_grid <- expand.grid(kmax=c(2, 3, 5),
                       distance=c(1, 2),
                       kernel=c("gaussian", "triangular"))

# To control the re-sampling
set.seed(825) 

# To train the model
fit1 <- train(Price ~ Weight + MPG.city, 
              data = train_data, 
              method = "kknn", 
              metric = "RMSE",
              trControl = fitControl,
              tuneGrid = my_grid)

# To show the results
fit1

# To plot the results
plot(fit1)

# To explore the best model
fit1$bestTune

# Using random grid -------------------------------------------------------
# Aqui vamos a dejar que train elija los valores de los 
# hiper-parametros (los que pueda elegir). Solo vamos a pedir
# que considere 4 valores

# To control the re-sampling
set.seed(825)

# To train the model
fit2 <- train(Price ~ Weight + MPG.city, 
              data = train_data, 
              method = "kknn", 
              metric = "RMSE",
              trControl = fitControl,
              tuneLength = 4)

# To show the results
fit2

# To plot the results
plot(fit2)

# To explore the best model
fit2$bestTune


# Comparing ---------------------------------------------------------------

# Here we are going to compare the fitted models

# Using the first model
y_hat1 <- predict(fit1, newdata=test_data, type="raw")
y_test <- test_data$Price
plot(x=y_test, y=y_hat1, las=1, pch=20)
abline(a=0, b=1, col="blue3")

# Para ver el ECM
mean((y_test - y_hat1)^2)

# Para ver la correlacion
cor(y_test, y_hat1)

# Using the second model
y_hat2 <- predict(fit2, newdata=test_data, type="raw")
y_test <- test_data$Price
plot(x=y_test, y=y_hat2, las=1, pch=20)
abline(a=0, b=1, col="blue3")

# Para ver el ECM
mean((y_test - y_hat2)^2)

# Para ver la correlacion
cor(y_test, y_hat2)

