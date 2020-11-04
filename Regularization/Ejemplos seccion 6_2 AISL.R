# -------------------------------------------------------------------------
# Este es el ejemplo de la seccion 6.2.1 pag 215
# de AISL de James et al. (2015)
# -------------------------------------------------------------------------

# Paquete asociado al libro AISL de James et al. (2015)
library(ISLR)
data("Hitters")

# Vamos a contar el numero de NA en cada columna de Hitters
apply(X=Hitters, MARGIN=2, function(x) sum(is.na(x)))

# Vemos que hay unos NA en la columna Salary
# Vamos a quitar esas observaciones
Hitters <- na.omit(Hitters)

# Vamos a alistar la informacion
x <- model.matrix(Salary ~ ., data=Hitters)[, -1]
y <- Hitters$Salary

# Aplicando ridge
library(glmnet)

# Vamos a considerar varios valores de lambda
grid <- 10^seq(10, -2, length=100)

# Vamos a ajustar el modelo
rigde.mod <- glmnet(x, y, alpha=0, lambda=grid,
                    family="gaussian", standardize=TRUE)

# Para ver los coeficientes estimados para cada lambda:
coef(rigde.mod)

# Para dibujar la evolucion de los coeficientes
plot(rigde.mod, label=TRUE, xvar="norm")
plot(rigde.mod, label=TRUE, xvar="lambda")
plot(rigde.mod, label=TRUE, xvar="dev")

# Para dibujar la evolucion de los coeficientes vs lambda
betas <- coef(rigde.mod)
matplot(betas, type='l')


# Aplicacion a base Credit ------------------------------------------------

# Vamos a alistar la informacion
x <- model.matrix(Balance ~ - 1 + Income + Limit + Rating + Student,
                  # + Cards + Age + Education + Gender + Married + Ethnicity, 
                  data=Credit)
y <- Credit$Balance

# Aplicando ridge
library(glmnet)

# Vamos a considerar varios valores de lambda
grid <- seq(from=0, to=10000, by=100)
grid <- c(0, 100, 1000, 10000)

# Vamos a ajustar el modelo
rigde.mod <- glmnet(x=x, y=y, 
                    alpha=0, lambda=grid,
                    standardize=FALSE,
                    family="gaussian")

# Para ver los coeficientes estimados para cada lambda:
coef(rigde.mod)

# Para dibujar la evolucion de los coeficientes
plot(rigde.mod)
plot(rigde.mod, label=TRUE, xvar="lambda", las=1)
plot(rigde.mod, label=TRUE, xvar="norm", las=1)
plot(rigde.mod, label=TRUE, xvar="dev", las=1)

# Para hacer mi propio dibujo
betas <- coef(rigde.mod)[-1, ]
matplot(y=t(betas), x=matrix(grid, ncol=1), type="l", las=1,
        xlab=expression(lambda), ylab=expression(beta))

mod0 <- lm(Balance ~ Income + Limit + Rating + Student, data=Credit)
coef(mod0)
