# En este ejemplo se muestra como ajustar un modelo lm
# usando el paquete caret

# Dataset
n <- 200
x1 <- rpois(n, lambda=5)
x2 <- rbinom(n, size=6, prob=0.4)
y <- rnorm(n, mean=-3+2*x1+4*x2, sd=2)
datos <- data.frame(y=y, x1=x1, x2=x2)
head(datos)

# Training the model
library(caret)

fitControl <- trainControl(method = "boot",
                           p=0.75, 
                           number = 15)

mod <- train(y ~ x1 + x2, 
             data = datos,
             method  = "lm",
             metric = "Rsquared",
             trControl = fitControl)

# Para ver las medidas de desempeno
mod$resample
# Para explorar las medidas
hist(mod$resample$RMSE, las=1)




# Explorando el objeto mod
mod
class(mod)
names(mod)
mod$bestTune
mod$results

mod$finalModel


plot(mod, metric="RMSE")
plot(mod, metric="Rsquared")
plot(mod, metric="MAE")
plot(mod, metric="RMSESD")
plot(mod, metric="RsquaredSD")

plot(mod, plotType = "level")
plot(mod, plotType = "scatter")

ggplot(mod) + theme_bw()

# Sacando el mejor modelo
fit <- mod$finalModel
fit
caret::R2(fit)



