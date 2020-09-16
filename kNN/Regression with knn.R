
# En este ejemplo vamos a utilizar la base de datos Cars93
# del paquete MASS para estimar el precio en funcion del
# peso y del rendimiento del combustible

library(FNN)

# Los datos que vamos a usar
library(MASS)
str(Cars93)

# Exploremos los datos
library(plotly)
Cars93 %>% plot_ly(x=~Weight, y=~MPG.city, z=~Price, 
                   color=~Price)


