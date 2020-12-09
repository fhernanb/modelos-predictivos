# -------------------------------------------------------------------------
# En este script se muestra como realizar Backwards Feature Selection
# con la funcion rfe (Recursive Feature Elimination) 
# del paquete caret.
# -------------------------------------------------------------------------


library(caret)
library(MASS)

# Opciones para functions -------------------------------------------------

# lmFuncs: linear regression
# nbFuncs: naive bayes
# treebagsFuncs: bagged tree
# rfFuncs: random forest
# caretFuncs: if model has tuning parameters that must be determined 
#             at each iteration.


# Los datos ---------------------------------------------------------------
datos <- na.omit(Cars93)

# Ejemplo regresion con covariables CUANTI --------------------------------
control <- rfeControl(functions=rfFuncs,
                      method="cv",
                      number=10)

my_form <- formula(Price ~ MPG.city + Horsepower + Weight +
                     Length + EngineSize + Passengers +
                     Luggage.room + Width)

results <- rfe(form=my_form,
               data=datos,
               size=1:8,
               #metric="Rsquared",
               metric="RMSE",
               rfeControl=control)

# summarize the results
print(results)
# list the chosen features
predictors(results)
# para ver el ajuste
results$fit
# plot the results
plot(results, type=c("g", "o"))
ggplot(results)

# Ejemplo regresion con covariables CUALI ---------------------------------
control <- rfeControl(functions=rfFuncs,
                      method="cv",
                      number=10)

my_form <- formula(Price ~ MPG.city + Horsepower + Weight +
                     Length + EngineSize + Passengers +
                     Luggage.room + Width + Type)

results <- rfe(form=my_form,
               data=datos,
               size=1:13,
               #metric="Rsquared",
               #metric="RMSE",
               rfeControl=control)

# summarize the results
print(results)
# list the chosen features
predictors(results)
# para ver el ajuste
results$fit
# plot the results
plot(results, type=c("g", "o"))


# Ejemplo clasificacion binaria -------------------------------------------

control <- rfeControl(functions=rfFuncs,
                      method="cv",
                      number=10)

my_form <- formula(Origin ~ MPG.city + Horsepower + Weight +
                     Length + EngineSize + Passengers +
                     Luggage.room + Width)

results <- rfe(form=my_form,
               data=datos,
               size=1:8,
               #metric="Accuracy",
               metric="Kappa",
               rfeControl=control)

# summarize the results
print(results)
# list the chosen features
predictors(results)
# para ver el ajuste
results$fit
# plot the results
plot(results, type=c("g", "o"))
ggplot(results)

