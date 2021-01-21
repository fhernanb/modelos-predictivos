# Este ejemplo corresponde al ejemplo mostrado en la vineta disponible en
# https://cran.r-project.org/web/packages/lime/index.html

# Libraries and data 
library(MASS)
library(lime)
data(biopsy)

# First we'll clean up the data a bit
biopsy$ID <- NULL
biopsy <- na.omit(biopsy)
names(biopsy) <- c('clump thickness', 'uniformity of cell size', 
                   'uniformity of cell shape', 'marginal adhesion',
                   'single epithelial cell size', 'bare nuclei', 
                   'bland chromatin', 'normal nucleoli', 'mitoses',
                   'class')

# Now we'll fit a linear discriminant model on all but 4 cases
set.seed(4)
test_set <- sample(seq_len(nrow(biopsy)), 4)
test_set <-  c(1, 2, 6, 13)
prediction <- biopsy$class
biopsy$class <- NULL
model <- lda(x=biopsy[-test_set, ], grouping=prediction[-test_set])

# If we use the model to predict the 4 remaining cases
# we get some pretty solid predictions:
predict(model, biopsy[test_set, ])

# But lets see how these predictions came to be, using lime.

explainer <- lime(x=biopsy[-test_set, ], 
                  model=model, 
                  bin_continuous = TRUE, 
                  quantile_bins = FALSE)

explanation <- explain(x=biopsy[test_set, ], 
                       explainer=explainer, 
                       n_labels = 1,
                       n_features = 4)

# Only showing part of output for better printing
explanation[, 2:9]

plot_features(explanation, ncol = 1)
