# En este ejemplo se muestra como estimar las medidas de
# desempeno
# https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/

# Dataset
n <- 100
x1 <- rpois(n, lambda=5)
x2 <- rbinom(n, size=6, prob=0.4)
y <- rnorm(n, mean=-3+2*x1+4*x2, sd=2)
datos <- data.frame(y=y, x1=x1, x2=x2)
head(datos)

# Bootstrap
library(caret)

fitControl <- trainControl(method = "boot", number = 200,
                           p = 0.75)

mod <- train(y ~ x1 + x2, 
             data = datos,
             method  = "lm",
             metric = "Rsquared",
             trControl = fitControl)

mod

# Para ver las medidas de desempeno
mod$resample

# Para explorar las medidas
par(mfrow=c(2, 2))
hist(mod$resample$RMSE, las=1)
hist(mod$resample$Rsquared, las=1)
hist(mod$resample$MAE, las=1)

# k-fold Cross Validation
library(caret)

fitControl <- trainControl(method="cv", number=10,
                           p = 0.75)

mod <- train(y ~ x1 + x2, 
             data = datos,
             method  = "lm",
             metric = "Rsquared",
             trControl = fitControl)

mod

# Para ver las medidas de desempeno
mod$resample

# Para explorar las medidas
par(mfrow=c(2, 2))
hist(mod$resample$RMSE, las=1)
hist(mod$resample$Rsquared, las=1)
hist(mod$resample$MAE, las=1)

# Repeated k-fold Cross Validation
library(caret)

fitControl <- trainControl(method="repeatedcv", 
                           number=10, 
                           repeats=5,
                           p = 0.75)

mod <- train(y ~ x1 + x2, 
             data = datos,
             method  = "lm",
             metric = "Rsquared",
             trControl = fitControl)

mod

# Para ver las medidas de desempeno
mod$resample

# Para explorar las medidas
par(mfrow=c(2, 2))
hist(mod$resample$RMSE, las=1)
hist(mod$resample$Rsquared, las=1)
hist(mod$resample$MAE, las=1)

# Leave One Out Cross Validation
library(caret)

fitControl <- trainControl(method="LOOCV")

mod <- train(y ~ x1 + x2, 
             data = datos,
             method  = "lm",
             metric = "Rsquared",
             trControl = fitControl)

mod


