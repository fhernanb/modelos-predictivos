# En este script se muestra como ajustar un modelo glm.nb
# de forma tradicional y usando caret

# Funcion para generar los datos
gen_dat <- function(n, b0, b1, k) {
  x <- runif(n=n)
  mu <- exp(b0 + b1 * x)
  y <- rnbinom(n=n, mu=mu, size=k)
  data.frame(y=y, x=x)
}

# Generando los datos
n <- 150
datos <- gen_dat(n=n, b0=-1, b1=2, k=3)

# Ajustado el modelo
library(MASS)
mod1 <- glm.nb(y ~ x, data=datos)
coef(mod1)

# usando caret
library(caret)

fitControl <- trainControl(method = "none")

fit <- train(y ~ x, data=datos,
             method="glm.nb",
             metric="RMSE",
             trControl = fitControl)

mod2 <- fit$finalModel
coef(mod2)

# usando caret pero con cv
fitControl <- trainControl(method = "repeatedcv", 
                           number = 1,
                           verboseIter = TRUE)

fit <- train(y ~ x, data=datos,
             method="glm.nb",
             metric="RMSE",
             #link="log",
             trControl=fitControl)


