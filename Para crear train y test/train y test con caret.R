# -------------------------------------------------------------------------
# En este script se muestra como crear los conjuntos de entrenamiento y
# de prueba para usar diversos modelos predictivos
# -------------------------------------------------------------------------


# Vamos a usar una base de datos pequena con 36 observaciones,
# a continuacion las instrucciones para leer los datos.
url <- 'https://tinyurl.com/k55nnlu'
datos <- read.table(file=url, header=T)

# Hay varias formas para crear train y test, aqui vamos a usar caret
library(caret)

# La funcion createDataPartition nos sirve para crear los indices
# que luego usaremos para dividir la base de datos.
# Consulte la ayuda de createDataPartition

set.seed(12345) # fijar semilla
trainIndex <- createDataPartition(y=datos$peso, 
                                  p=0.8, # 80% train y 20% test
                                  list=FALSE, 
                                  times=1)

# Para ver los indices que identifican las observaciones de train
trainIndex

# Para crear train y test
datos_train <- datos[ trainIndex, ]
datos_test  <- datos[-trainIndex, ]

# Para explorar train y test
datos_train
datos_test
