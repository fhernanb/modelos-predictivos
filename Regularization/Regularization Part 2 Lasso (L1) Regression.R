# -------------------------------------------------------------------------
# Los ejemplos mostrados aqui estan basados en el video
# Regularization Part 2: Lasso (L1) Regression
# https://youtu.be/NGf0voTMlcs
# -------------------------------------------------------------------------


# Los datos que vamos a usar
library(MASS)
str(Cars93)


# Ejemplo 1 ---------------------------------------------------------------
# En este ejemplo se creara un modelo de regresion
# lineal simple para explicar Precio ~ EngineSize

# Para ver los datos
with(Cars93, plot(x=EngineSize, y=Price))

# Supongamos que nuestro conjunto de Train es pequeno
# y que solo tiene 3 observaciones
indices <- c(84, 40)
train_data <- Cars93[ indices, c("Price", "EngineSize")]
test_data  <- Cars93[-indices, c("Price", "EngineSize")]

# Vamos a ver las observaciones seleccionadas para Train
with(Cars93, plot(x=EngineSize, y=Price))
with(train_data, points(x=EngineSize, y=Price, col="tomato", pch=19))

# Funcion objetivo a minimizar, Minimos Cuadrados
func_obj <- function(betas, data) {
  b0 <- betas[1] # intercepto
  b1 <- betas[2] # pendiente
  y_hat <- b0 + b1 * data$EngineSize
  ei <- data$Price - y_hat
  return(sum(ei^2))
}

# Usemos optim para encontrar los valores de b0 y b1
# que minimizan la funcion objetivo
mod1 <- optim(par=c(0, 0), fn=func_obj, data=train_data)
mod1

# Vamos a calcular el Error Cuadratico Medio (ECM) 
# con Train y Test usando el modelo mod creado antes

# ECM manual para Train
y_hat <- mod1$par[1] + mod1$par[2] * train_data$EngineSize
ecm_train1 <- mean((train_data$Price - y_hat)^2)
ecm_train1

# ECM manual para Test
y_hat <- mod1$par[1] + mod1$par[2] * test_data$EngineSize
ecm_test1 <- mean((test_data$Price - y_hat)^2)
ecm_test1

# Funcion objetivo a minimizar usando LASSO (l)
func_obj_l <- function(betas, l, data) {
  b0 <- betas[1] # intercepto
  b1 <- betas[2] # pendiente
  y_hat <- b0 + b1 * data$EngineSize
  ei <- data$Price - y_hat
  return(sum(ei^2) + l * abs(b1))
}

# Usemos optim para encontrar los valores de b0 y b1
# que minimizan la funcion objetivo

l <- 10 # valor de penalizacion lambda (seleccion caprichosa)
mod2 <- optim(par=c(0, 0), fn=func_obj_l, l=l, data=train_data)
mod2

# Vamos a calcular el Error Cuadratico Medio (ECM) 
# con Train y Test usando el modelo mod creado antes

# ECM manual para Train
y_hat <- mod2$par[1] + mod2$par[2] * train_data$EngineSize
ecm_train2 <- mean((train_data$Price - y_hat)^2)
ecm_train2

# ECM manual para Test
y_hat <- mod2$par[1] + mod2$par[2] * test_data$EngineSize
ecm_test2 <- mean((test_data$Price - y_hat)^2)
ecm_test2

# Comparando los resultados
cbind(ecm_train1, ecm_test1, ecm_train2, ecm_test2)

# Ejemplo 2 ---------------------------------------------------------------
# En este ejemplo se creara un modelo de regresion 
# lineal LASSO usando el paquete glmnet para explicar 
# Price ~ EngineSize + Weight + MPG.city

# Ajustando el modelo con lm
fit <- lm(Price ~ EngineSize + Weight + MPG.city, 
          data = Cars93)
coef(fit)

# Ajustando el modelo con lasso manual
func_obj_l <- function(betas, l, data) {
  b0 <- betas[1] # intercepto
  b1 <- betas[2] # efecto de enginesize
  b2 <- betas[3] # efecto de weight
  b3 <- betas[4] # efecto de mpgcity
  y_hat <- b0 + b1 * data$EngineSize + b2 * data$Weight + b3 * data$MPG.city
  ei <- data$Price - y_hat
  return(sum(ei^2) + l * (abs(b1) + abs(b2) + abs(b3)))
}

my_lasso <- optim(par=c(0, 0, 0, 0), fn=func_obj_l, 
                  l=0, data=Cars93, 
                  control=list(maxit=10000, reltol=1e-15))

my_lasso$par # coincide con coef(fit)

# Aplicando LASSO regression de forma automatica con glmnet
# visitar este enlace para mas detalles: https://rpubs.com/Joaquin_AR/242707
library(glmnet)

y <- Cars93$Price
x <- model.matrix(Price ~ - 1 + EngineSize + Weight + MPG.city, 
                  data = Cars93)

# Para explorar la matriz x
head(x)

# Para obtener un ajuste mediante lasso regression 
# se usa argumento alpha=1.
modelos_lasso <- glmnet(x=x, y=y, alpha=1,
                        family="gaussian")

# Para ver los resultados
plot(modelos_lasso, xvar = "lambda", label = TRUE, las=1)
grid()

# Con el fin de identificar el valor de lambda que da lugar al 
# mejor modelo, se puede recurrir a Cross-Validation. 
# La funcion cv.glmnet() calcula el cv-test-error, 
# utilizando por defecto k=10.

set.seed(1)
# x e y son la matriz modelo y el vector respuesta creados anteriormente
cv_error_lasso <- cv.glmnet(x=x, y=y, alpha=1, nfolds=10,
                            type.measure="mse")

# Para ver los resultados
plot(cv_error_lasso, las=1)
grid()

# El grafico muestra el cv-test-error (Mean Square Error) para 
# cada valor de lambda junto con la barra de error correspondiente. 
# Entre la informacion almacenada en el objeto devuelto por la 
# funcion cv.glmnet() se encuentra el valor de lambda con el que se 
# consigue el menor cv-test error y el valor de lambda con el que 
# se consigue el modelo mas sencillo que se aleja menos de 1 
# desviacion estandar del minimo cv-test-error posible.

# Valor lambda con el que se consigue el minimo test-error
cv_error_lasso$lambda.min

# Valor lambda optimo: mayor valor de lambda con el que el 
# test-error no se aleja mas de 1 sd del minimo test-error posible.
cv_error_lasso$lambda.1se

# Se muestra el valor de los coeficientes para el valor de 
# lambda optimo
modelo_final_lasso <- glmnet(x=x, y=y, alpha=1, 
                             lambda=cv_error_lasso$lambda.min)
coef(modelo_final_lasso)

# Explorando las predicciones
y_hat <- predict(modelo_final_lasso, newx=x)

# mse
mean((y - y_hat)^2)

# rho(y, y_hat)
cor(y, y_hat)

# Diagrama de dispersion entre y e y_hat
plot(x=y, y=y_hat, las=1, asp=1)
abline(a=0, b=1, lty="dashed", col="tomato")
