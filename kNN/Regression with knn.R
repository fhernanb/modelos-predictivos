
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
mod <- knn.reg(train=x_train, test=x_test, y=y_train, 
               k=5, algorithm="kd_tree")

# Para ver la clase del objeto mod
class(mod)

# Para ver los elementos dentro de mod
names(mod)

# Para explorar $pred
y_hat <- mod$pred

# Para ver el ECM
mean((y_test - y_hat)^2)

# Para ver la correlacion
cor(y_test, y_hat)

# Para ver al relacion
plot(x=y_test, y=y_hat, las=1)
abline(a=0, b=1, col="blue3")
