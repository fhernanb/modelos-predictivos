# -------------------------------------------------------------------------
# En este ejemplo vamos a utilizar la base de datos titanic
# para predecir si una persona sobrevive o no usando como covariables
# Pclass, Sex, Age y Fare.
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

# Valor de log-verosimilitud
logLik(mod1)



