# -------------------------------------------------------------------------
# Why is glmnet ridge regression giving me a different answer 
# than manual calculation?
# -------------------------------------------------------------------------
# 
# Esta pregunta fue hecha en:
# https://stats.stackexchange.com/questions/129179/why-is-glmnet-ridge-regression-giving-me-a-different-answer-than-manual-calculat

# I'm using glmnet to calculate ridge regression estimates. 
# I got some results that made me suspicious in that glmnet is 
# really doing what I think it does. To check this I wrote 
# a simple R script where I compare the result of ridge regression 
# done by solve and the one in glmnet, the difference is significant:

n    <- 1000
p    <-  100
X    <- matrix(rnorm(n*p, 0, 1), n, p)
beta <- rnorm(p, 0, 1)
Y    <- X%*%beta + rnorm(n,0,0.5)

beta1 <- solve(t(X) %*% X + 5*diag(p), t(X) %*% Y) # manual
beta2 <- glmnet(X,Y, alpha=0, lambda=10, 
                intercept=FALSE, 
                standardize=FALSE, 
                family="gaussian")$beta@x
beta1 - beta2

# The norm of the difference is usually around 20 which cannot 
# be due to numerically different algorithms, I must be doing 
# something wrong. What are the settings I have to set in glmnet 
# in order to obtain the same result as with ridge?

# A continuacion la primera respuesta

library(MASS)
datos <- Cars93[, c("Price", "EngineSize", "Weight", "MPG.city")]
               
# Escalando las x's
#datos[, 2:4] <- scale(datos[, 2:4])

# La desviacion de y
y <- datos$Price
n <- length(y)
sd_y <- sqrt(var(y)*(n-1)/n)


# Ajustando el modelo con Ridge manual
func_obj_l <- function(betas, l, data) {
  b0 <- betas[1] # intercepto
  b1 <- betas[2] # efecto de enginesize
  b2 <- betas[3] # efecto de weight
  b3 <- betas[4] # efecto de mpgcity
  y_hat <- b0 + b1 * data$EngineSize + b2 * data$Weight + b3 * data$MPG.city
  ei <- data$Price - y_hat
  return(sum(ei^2) / (2*n) + l * (b1^2 + b2^2 + b3^2) / (2*sd_y))
}

my_ridge <- optim(par=c(0, 0, 0, 0), fn=func_obj_l, 
                  l=9.26, data=datos, 
                  control=list(maxit=10000, reltol=1e-15))

my_ridge$par
my_ridge$par * sd_y
my_ridge$par * sd_y^2


library(glmnet)

# Debemos crear la matriz x y el vector y
y <- Cars93$Price
x <- model.matrix(Price ~ - 1 + EngineSize + Weight + MPG.city, 
                  data=Cars93)

mod <- glmnet(x=x, y=y,
              alpha=0, lambda=9.26,
              intercept=TRUE,
              family="gaussian")

# Para ver los coeficientes
coef(mod) # valores cercanos a los obtenidos manualmente!!!

my_ridge$par
