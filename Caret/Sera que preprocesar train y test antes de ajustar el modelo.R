library(caret)

# Dataset
n <- 1000
x1 <- rpois(n, lambda=5)
x2 <- rbinom(n, size=6, prob=0.4)
y <- rnorm(n, mean=-3+2*x1+4*x2, sd=2)
datos <- data.frame(y=y, x1=x1, x2=x2)
head(datos)

# To select some observations
inTrain <- sample(seq(along = datos$y), length(datos$y)/2)

training <- datos[inTrain,]
test <- datos[-inTrain,]

# Preprocessing
preProcValues <- preProcess(training, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, test)

# Training the model
mod <- train(y ~ x1 + x2, 
             data = training,
             method  = "lm",
             preProcess = c("center", "scale"))

mod$bestTune
fit <- mod$finalModel
fit

par(mfrow=c(2, 2))

# Caso 1
y_hat <- predict(fit, newdata = training, type = "response")
y_true <- training$y
plot(x=y_true, y=y_hat, main=cor(y_true, y_hat))
abline(a=0, b=1, col = "tomato")

# Caso 2
y_hat <- predict(fit, newdata = trainTransformed, type = "response")
y_true <- training$y
plot(x=y_true, y=y_hat, main=cor(y_true, y_hat))
abline(a=0, b=1, col = "tomato")

# Caso 3
y_hat <- predict(fit, newdata = test, type = "response")
y_true <- test$y
plot(x=y_true, y=y_hat, main=cor(y_true, y_hat))
abline(a=0, b=1, col = "tomato")

# Caso 4
y_hat <- predict(fit, newdata = testTransformed, type = "response")
y_true <- test$y
plot(x=y_true, y=y_hat, main=cor(y_true, y_hat))
abline(a=0, b=1, col = "tomato")




