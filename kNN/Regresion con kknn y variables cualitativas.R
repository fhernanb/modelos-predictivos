# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar la base de datos Cars93
# del paquete MASS para estimar el precio del auto en funcion del
# AirBags, DriveTrain y Origin
# -------------------------------------------------------------------------

# Los datos que vamos a usar
library(MASS)
head(Cars93)

# Vamos a explorar los datos
library(tidyverse)
Cars93 %>% glimpse()

# Exploremos la relacion entre Y y las covariables
with(Cars93, plot(Price ~ AirBags))
with(Cars93, plot(Price ~ DriveTrain))
with(Cars93, plot(Price ~ Origin))

# Particion de los datos, vamos a usar aprox 60% y 40% para train y test
i_train <- sample(1:93, size=60)

train_data <- Cars93[i_train, ] # 60 obs
test_data <- Cars93[-i_train, ] # 33 obs

# Usando el paquete kknn --------------------------------------------------
library(kknn)

# train.kknn sirve para ajustar el modelo y 
# sirve encontrar los hiperparametros simulataneamente.
fit1 <- train.kknn(Price ~ AirBags + DriveTrain + Origin,
                   data=train_data,
                   distance=3,
                   kmax=2,
                   kernel="gaussian",
                   scale=TRUE)

# Para ver la clase del objeto
class(fit1)

# Para ver los elementos dentro del objeto
names(fit1)


# Explorando el desempeno usando train ------------------------------------
y_true_train <- train_data$Price

# Para explorar las estimaciones
y_hat1 <- predict(fit1, newdata=train_data)

# Para ver el ECM
mean((y_true_train - y_hat1)^2)

# Para ver la correlacion
cor(y_true_train, y_hat1)

# Para ver la relacion entre y_test y y_hat
plot(x=y_true_train, y=y_hat1, las=1, pch=20)
abline(a=0, b=1, col="blue3")

# Explorando el desempeno usando test ------------------------------------
y_true_test <- test_data$Price

# Para explorar las estimaciones
y_hat2 <- predict(fit1, newdata=test_data)

# Para ver el ECM
mean((y_true_test - y_hat2)^2)

# Para ver la correlacion
cor(y_true_test, y_hat2)

# Para ver la relacion entre y_test y y_hat
plot(x=y_true_test, y=y_hat2, las=1, pch=20)
abline(a=0, b=1, col="blue3")

# -------------------------------------------------------------------------
# Homework:
# Busque los valores de los hiper-parametros para mejorar el modelo
# -------------------------------------------------------------------------
