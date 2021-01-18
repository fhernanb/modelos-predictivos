# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar la base de datos titanic
# para predecir si una persona sobrevive o no usando como covariables
# Pclass, Sex, Age y Fare.
#
# Con el modelo entrenado vamos a ilustrar la interpretacion
# de los coeficientes estimados
# -------------------------------------------------------------------------

# Los datos a usar estan disponibles en un repositorio de github
library(readr)

file <- "https://raw.githubusercontent.com/fhernanb/datos/master/titanic.csv"
datos <- read_csv(file)
head(datos)

# Explorando los datos
str(datos)

# Convirtiendo Pclass a factor
datos$Pclass <- as.factor(datos$Pclass)

# Entrenado el modelo
mod1 <- glm(Survived ~ Pclass + Sex + Age + Fare,
            data=datos, family=binomial(logit))

summary(mod1)

# Volvamos a ajustar el modelo sin Fare
mod2 <- update(mod1, . ~ . - Fare)

summary(mod2)


# Ilustracion 1 -----------------------------------------------------------

# Supongamos que tenemos dos personas similares en casi todo, B tiene un
# ano mas de edad que A

dt <- data.frame(Pclass=as.factor(c(2, 2)),
                 Sex=c("male", "male"),
                 Age=c(29,30),
                 row.names=c("A", "B"))

dt

# Vamos a calcular las probabilidades de sobrevivir para A y B
probs <- predict(object=mod2, newdata=dt, type='response')
probs

# Separando las probabilidades
p_a <- probs[1]
p_b <- probs[2]

# Restando las probabilidades, sera eso?
p_b - p_a

# Comparando los odds
p_b / (1 - p_b)
p_a / (1 - p_a)

# Restando los odds
(p_b / (1 - p_b)) - (p_a / (1 - p_a))

# Dividiendo los odds
(p_b / (1 - p_b)) / (p_a / (1 - p_a))

# Comparando logit(pi) = b0 + b1 xi
pred_lineal <- predict(object=mod2, newdata=dt, type='link')
pred_lineal

diff(pred_lineal)



