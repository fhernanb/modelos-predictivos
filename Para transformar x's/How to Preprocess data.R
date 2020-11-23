# -------------------------------------------------------------------------
# En este archivo se muestran ejemplos de como preprocesar los datos
# antes de crear un modelo
# -------------------------------------------------------------------------

library(caret)

# Vamos a generar n valores para crear una base de datos artificial
n <- 1000
x1 <- rpois(n, lambda=5)
x2 <- rbinom(n, size=6, prob=0.4)
y <- rnorm(n, mean=-3+2*x1+4*x2, sd=3)
datos <- data.frame(y=y, x1=x1, x2=x2)
head(datos)

# To select some observations
inTrain <- sample(seq(along = datos$y), length(datos$y)/2)

dat_train <- datos[ inTrain, ]
dat_test  <- datos[-inTrain, ]

# Explorando con graficos
par(mfrow=c(1, 2))
boxplot(dat_train, las=1, main="Train")
boxplot(dat_test, las=1, main="Test")

# Preprocessing to 0 - 1 --------------------------------------------------
preProcValues <- preProcess(dat_train, method="range")
dat_train_transf <- predict(preProcValues, dat_train)
dat_test_transf  <- predict(preProcValues, dat_test)

# Explorando con graficos
boxplot(dat_train_transf, las=1, main="Train transf")
boxplot(dat_test_transf, las=1, main="Test transf")

# Vamos a explorar la media y varianza de los datos sin/con transformacion
# pero vamos a crear una funcion para esto.
funcioncita <- function(x) c(Minimo=min(x), 
                             Media=mean(x), Mediana=median(x),
                             Desvi=sd(x), Vari=var(x), 
                             Maximo=max(x))

apply(X=dat_train, MARGIN=2, FUN=funcioncita)
apply(X=dat_train_transf, MARGIN=2, FUN=funcioncita)
apply(X=dat_test_transf, MARGIN=2, FUN=funcioncita) 
# Por que no coinciden los resultados?

# Preprocessing to -3 to 3 ------------------------------------------------
preProcValues <- preProcess(dat_train, method=c("center", "scale"))
dat_train_transf <- predict(preProcValues, dat_train)
dat_test_transf  <- predict(preProcValues, dat_test)

# Explorando con graficos
boxplot(dat_train_transf, las=1, main="Train transf")
boxplot(dat_test_transf, las=1, main="Test transf")

# Vamos a explorar la media y varianza de los datos sin/con transformacion
apply(X=dat_train, MARGIN=2, FUN=funcioncita)
apply(X=dat_train_transf, MARGIN=2, FUN=funcioncita)
apply(X=dat_test_transf, MARGIN=2, FUN=funcioncita) 

# Preprocessing with knnImpute --------------------------------------------

# Recordemos los datos
head(dat_train)

# Vamos a incluir 3 NA's
dat_train[1, 1] <- NA
dat_train[2, 2] <- NA
dat_train[3, 3] <- NA
head(dat_train)

# usando range
preProcValues <- preProcess(dat_train, method="range")
dat_train_transf <- predict(preProcValues, dat_train)
head(dat_train_transf)

# usando range y knnImpute
preProcValues <- preProcess(dat_train, method=c("range", "knnImpute"))
dat_train_transf <- predict(preProcValues, dat_train)
head(dat_train_transf)

