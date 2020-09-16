library(caret)

# Dataset
n <- 10000
x1 <- rpois(n, lambda=5)
x2 <- rbinom(n, size=6, prob=0.4)
y <- rnorm(n, mean=-3+2*x1+4*x2, sd=3)
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

# Comparing
head(training)
head(trainTransformed)

head(test)
head(testTransformed)

# Verifying
colMeans(trainTransformed)
var(trainTransformed)

colMeans(testTransformed)
var(testTransformed)

# Scaling to 0 - 1
preProcValues <- preProcess(training, method = "range")
a <- predict(preProcValues, training)
b <- predict(preProcValues, test)
head(a)
head(b)

