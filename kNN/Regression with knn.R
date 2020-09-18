
# En este ejemplo vamos a utilizar la base de datos Cars93
# del paquete MASS para estimar el precio en funcion del
# peso y del rendimiento del combustible

# Los datos que vamos a usar
library(MASS)
str(Cars93)

# Exploremos los datos
library(plotly)
Cars93 %>% plot_ly(x=~Weight, y=~MPG.city, z=~Price, color=~Price)

# Preparando los datos
indices <- sample(1:93, size=60)

x_train <- Cars93[indices, c("Weight", "MPG.city")]
y_train <- Cars93[indices, "Price"]

x_test <- Cars93[-indices, c("Weight", "MPG.city")]
y_test <- Cars93[-indices, "Price"]


# Usando el paquete FNN ---------------------------------------------------
library(FNN)

fit1 <- knn.reg(train=x_train, test=x_test, y=y_train, 
               k=5, algorithm="kd_tree")

# Para ver la clase del objeto
class(fit1)

# Para ver los elementos dentro del objeto
names(fit1)

# Para explorar $pred
y_hat <- fit1$pred

# Para ver el ECM
mean((y_test - y_hat)^2)

# Para ver la correlacion
cor(y_test, y_hat)

# Para ver la relacion
plot(x=y_test, y=y_hat, las=1)
abline(a=0, b=1, col="blue3")


# Usando el paquete kknn --------------------------------------------------
library(kknn)

fit2 <- train.kknn(Price ~ Weight + MPG.city,
                   data=Cars93[indices, ],
                   distance=2,
                   kernel="triangular",
                   kmax=15,
                   kcv=10,
                   scale=FALSE)

# Para ver la clase del objeto
class(fit2)

# Para ver los elementos dentro del objeto
names(fit2)

# Para ver los mejora hyper parametros
fit2$best.parameters

# Para explorar las estimaciones
y_hat <- predict(fit2, newdata=Cars93[-indices, ])

# Para ver el ECM
mean((y_test - y_hat)^2)

# Para ver la correlacion
cor(y_test, y_hat)

# Para ver la relacion
plot(x=y_test, y=y_hat, las=1)
abline(a=0, b=1, col="blue3")




