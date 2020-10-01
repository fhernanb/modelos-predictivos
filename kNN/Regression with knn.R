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

Cars93 %>% select(Weight, MPG.city) %>% slice(i_train) -> x_train
Cars93 %>% select(Price) %>% slice(i_train) %>% pull() -> y_train

Cars93 %>% select(Weight, MPG.city) %>% slice(-i_train) -> x_test
Cars93 %>% select(Price) %>% slice(-i_train) %>% pull() -> y_test

# Otra forma de hacer lo anterior es:
# x_train <- Cars93[i_train, c("Weight", "MPG.city")]
# y_train <- Cars93[i_train, "Price"]
# x_test <- Cars93[-i_train, c("Weight", "MPG.city")]
# y_test <- Cars93[-i_train, "Price"]


# Usando el paquete FNN ---------------------------------------------------
library(FNN)

fit1 <- knn.reg(train=x_train, test=x_test, y=y_train, 
               k=5, algorithm="kd_tree")

# Para ver la clase del objeto
class(fit1)

# Para ver los elementos dentro del objeto
names(fit1)

# Para explorar $pred
y_hat1 <- fit1$pred

# Para ver el ECM
mean((y_test - y_hat1)^2)

# Para ver la correlacion
cor(y_test, y_hat1)

# Para ver la relacion entre y_test y y_hat
plot(x=y_test, y=y_hat1, las=1, pch=20)
abline(a=0, b=1, col="blue3")


# Usando el paquete kknn --------------------------------------------------
library(kknn)

fit2 <- train.kknn(Price ~ Weight + MPG.city,
                   data=Cars93[i_train, ],
                   distance=2,
                   kernel="triangular",
                   kmax=15,
                   kcv=10,
                   scale=FALSE)

# Para ver la clase del objeto
class(fit2)

# Para ver los elementos dentro del objeto
names(fit2)

# Para ver los mejores hyper parametros
fit2$best.parameters

# Para explorar las estimaciones
y_hat2 <- predict(fit2, newdata=x_test)

# Para ver el ECM
mean((y_test - y_hat2)^2)

# Para ver la correlacion
cor(y_test, y_hat2)

# Para ver la relacion entre y_test y y_hat
plot(x=y_test, y=y_hat2, las=1, pch=20)
abline(a=0, b=1, col="blue3")

