# -------------------------------------------------------------------------
# En este ejemplo se muestra como usar nn para regresion.
# El ejemplo esta basado en http://uc-r.github.io/ann_regression
# Los datos del ejemplo se van a simular pero tambien estan disponibles
# en un repositorio de github.
# -------------------------------------------------------------------------

# Simulando los datos -----------------------------------------------------
# Vamos a usar datos simulados de un modelo
# y ~ N(mu=4 - 3 * x1 + 3 * x2, sigma=6)

gen_dat <- function(n) {
  x1 <- runif(n=n, min=-5, max=6)
  x2 <- runif(n=n, min=-4, max=5)
  media <- 4 - 3 * x1 + 3 * x2
  y <- rnorm(n=n, mean=media, sd=6)
  marco_datos <- data.frame(y=y, x1=x1, x2=x2)
  return(marco_datos)
}

set.seed(1974)
datos <- gen_dat(n=10000)
head(datos)

# Los datos simulados estan disponibles tambien en la url de abajo:
url <- "https://raw.githubusercontent.com/fhernanb/datos/master/datos_regresion_mult_redes.txt"
datos <- read.table(url, header=TRUE)
head(datos)

# Visualizando los datos --------------------------------------------------
library(scatterplot3d)
scatterplot3d(x=datos$x1, y=datos$x2, z=datos$y, 
              pch=16, cex.lab=1,
              highlight.3d=TRUE, type="h", xlab="x1",
              ylab="x2", zlab="y")

# Creando la red neuronal con tidymodels ----------------------------------

library(tidymodels)

set.seed(123)
nn_fit <- mlp(hidden_units = 5,
              penalty = 0.01,
              epochs = 100,
              learn_rate = 0.01) |>
  set_engine("nnet") |>
  set_mode("regression") |>
  fit(y ~ ., data = datos)

pred <- predict(nn_fit, datos)
results <- bind_cols(datos, pred)
results |> head()

with(results, plot(x=y, y=.pred))
abline(a=0, b=1, col="tomato")

# Evaluate performance
metrics(results, truth = y, estimate = .pred)


