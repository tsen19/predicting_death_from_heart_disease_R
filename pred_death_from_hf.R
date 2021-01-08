###################################################################################################
## Script name: predicting_death_from_heart_failure.R
## Project: HarvardX PH125.9x | Data Science: Capstone
## Script purpose: Train and tune ML models to predict death from heart failure
## Date: January, 2021
## Author: Neli Tsereteli
## Contact information: 
## Notes: 
###################################################################################################

#########################################
# Set-up the environment                 #
#########################################

# Install missing packages automatically
if(!require(tidyverse)) install.packages("tidyverse",repos ="http://cran.us.r-project.org")
if(!require(caret))install.packages("caret",repos ="http://cran.us.r-project.org")
if(!require(caretEnsemble))install.packages("caretEnsemble",repos ="http://cran.us.r-project.org")
if(!require(data.table))install.packages("data.table",repos ="http://cran.us.r-project.org")
if(!require(stringr))install.packages("stringr",repos ="http://cran.us.r-project.org")
if(!require(lubridate))install.packages("lubridate",repos ="http://cran.us.r-project.org")
if(!require(knitr))install.packages("knitr",repos ="http://cran.us.r-project.org")
if(!require(rio))install.packages("rio",repos ="http://cran.us.r-project.org")
if(!require(dplyr))install.packages("dplyr",repos ="http://cran.us.r-project.org")
if(!require(skimr))install.packages("skimr",repos ="http://cran.us.r-project.org")
if(!require(PerformanceAnalytics))install.packages("PerformanceAnalytics",repos ="http://cran.us.r-project.org")

# Load the packages
library(tidyverse)    # a set of packages that work in harmony
library(caret)        # functions for training and plotting classification and regression models
library(caretEnsemble)# ensembles of caret models: caretList() and caretStack()
library(data.table)   # extension of 'data.frame'
library(stringr)      # simple, consistent wrappers for common string operations
library(lubridate)    # functions to work with date-times and time-spans
library(knitr)        # general-purpose tool for dynamic report generation in R
library(rio)          # a Swiss-Army Knife for Data I/O
library(dplyr)        # a grammar of data manipulation
library(skimr)        # compact and flexible summaries of data
library(PerformanceAnalytics) # econometric functions for performance

# Set plot options
knitr::opts_chunk$set(fig.width = 6, fig.height = 4) 

#############################################
# Data exploration and pre-processing       #
#############################################

# Load the data
heart_data <- rio::import("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

# Structure and descriptions
str(heart_data)
skimmed <- skim(heart_data)
skimmed[, c(2, 3, 7, 9, 11:12)]
rm(skimmed)

# Correlation plot
chart.Correlation(heart_data[1:12], histogram=TRUE, pch=19)

# Data pre-processing
# Factorize the outcome
heart_data$DEATH_EVENT <- as.factor(ifelse(heart_data$DEATH_EVENT == 1, "Died", "Survived"))

# Scale numerical
heart_data[, c("age", "creatinine_phosphokinase", "ejection_fraction",
               "platelets", "serum_creatinine", "serum_sodium", "time")] <- 
  scale(heart_data[, c("age", "creatinine_phosphokinase", "ejection_fraction",
               "platelets", "serum_creatinine", "serum_sodium", "time")])

# Create train-test partitions
# Test set will be 25% of the data
set.seed(1234)
test_index <- createDataPartition(y = heart_data$DEATH_EVENT, times = 1, p = 0.25, list = FALSE)
train <- heart_data[-test_index,]
test <- heart_data[test_index,]
rm(test_index)

# Save outcomes
train_outcome <- train %>% select(DEATH_EVENT)
test_outcome <- test %>% select(DEATH_EVENT)

#########################################
# Feature importance                    #
#########################################

# Visualize importance of variables with box plots
# caret's featurePlot
featurePlot(x = train[, 1:12], 
            y = train$DEATH_EVENT, 
            plot = "box", 
            strip = strip.custom(par.strip.text = list(cex = 0.7)),
            scales = list(x = list(relation = "free"), y = list(relation = "free")))

# Visualize importance of variables with density plots
featurePlot(x = train[, 1:12], 
            y = train$DEATH_EVENT, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), y = list(relation="free")))

# Feature selection using recursive feature elimination (RFE)
# Model sizes to consider
subsets <- c(1:11)

# Control: 
# k-fold cross validation repeated 3 times
# Random forest based rfFuncs
control <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 3,
                   verbose = FALSE)

# Recursive feature elimination
# Exclude outcome and time
set.seed(1234)
rfe_result <- rfe(x = train[, 1:11], y = train$DEATH_EVENT,
                 sizes = subsets,
                 rfeControl = control)
# Print result
rfe_result

# Subset the columns
train <- train %>% select(-c(sex, smoking, creatinine_phosphokinase, diabetes, time))
test <- test %>% select(-c(sex, smoking, creatinine_phosphokinase, diabetes, time))

############################################
# Model building - train control           #
############################################

# See available algorithms in caret
head(names(getModelInfo()))

# Train control
# Set 15-fold CV
# twoClassSummary because the outcome is binary
# Generate probabilities instead of classes
control <- trainControl(method = "cv", 
                        number = 15, 
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

#############################################
# Model building - KNN                      #
#############################################

# What properties and/or hyperparameters does knn have?
modelLookup("knn")

# Number of possible Ks to evaluate
possible_ks <- 25

# Train the model
set.seed(4242)
model_knn <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "knn", 
                   trControl = control, 
                   tuneLength = possible_ks,
                   metric = "Sens")

# Output and plot
model_knn
plot(model_knn, main = "Model sensitivities with KNN")

# Print best parameter
(best_k <- model_knn$bestTune)

# Predictions and confusion matrix
predicted <- predict(model_knn, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

# Feature importance
plot(varImp(model_knn), main = "Variable Importance with KNN")

###################################################
# Model building - RF                             #
###################################################

# What properties and/or hyperparameters does rf have?
modelLookup("rf")

# Number of possible mtrys to evaluate
possible_mtry <- 25

# Train the model
set.seed(4242)
model_rf <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "rf", 
                   trControl = control,
                   tuneLength = possible_mtry,
                   metric = "Sens")

# Output and plot
model_rf
plot(model_rf, main = "Model sensitivities with RF")

# Print best parameter
(best_mtry <- model_rf$bestTune)

# Predictions and confusion matrix
predicted <- predict(model_rf, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

# Variale importance
var_imp <- varImp(model_rf)
plot(var_imp, main = "Variable Importance with RF")


###################################################
# Model building - adaboost                       #
###################################################

# What properties and/or hyperparameters does adaboost have?
modelLookup("adaboost")

# Number of possible unique hyperparameters to evaluate
possible_params <- 3

# Train the model
set.seed(4242)
model_adaboost <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "adaboost", 
                   trControl = control,
                   tuneLength = possible_params,
                   metric = "Sens")

# Output and plot
model_adaboost
plot(model_adaboost)

# Print best parameter
(best <- model_adaboost$bestTune)

# Predictions and confusion matrix
predicted <- predict(model_adaboost, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

# Feature importance
var_imp <- varImp(model_adaboost)
plot(var_imp, main = "Variable Importance with adaboost")


###################################################
# Model building - LDA                            #
###################################################

# Train the model
set.seed(4242)
model_lda <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "lda", 
                   trControl = control,
                   metric = "Sens"
                   )

# Output of lda model
model_lda

## Predictions and confusion matrix
predicted <- predict(model_lda, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

## Feature importance
var_imp <- varImp(model_lda)
plot(var_imp, main = "Variable Importance with adaboost")


###################################################
# Model building - SVM linear                     #
###################################################

# What properties and/or hyperparameters does SVM linear have?
modelLookup("svmLinear")

# Number of possible unique hyperparameters to evaluate
possible_params <- 3

# Train the model
set.seed(4242)
model_svm <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "svmLinear", 
                   trControl = control,
                   tuneLength = possible_params,
                   metric = "Sens")

# Output of svm model
model_svm

# Predictions and confusion matrix
predicted <- predict(model_svm, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

# Feature importance
var_imp <- varImp(model_svm)
plot(var_imp, main = "Variable Importance with adaboost")

###################################################
# Model building - SVM radial                     #
###################################################

# What properties and/or hyperparameters does SVM radial have?
modelLookup("svmRadial")

# Number of possible unique hyperparameters to evaluate
possible_params <- 3

# Train the model
set.seed(4242)
model_svm_radial <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "svmRadial", 
                   trControl = control,
                   tuneLength = possible_params,
                   metric = "Sens")

# Output and plot
model_svm_radial
plot(model_svm_radial)

# Print best parameter
(best <- model_svm_radial$bestTune)

# Predictions and confusion matrix
predicted <- predict(model_svm_radial, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

# Feature importance
var_imp <- varImp(model_svm_radial)
plot(var_imp, main = "Variable Importance with SVM Radial")


###################################################
# Model building - GLMNET                         #
###################################################

# What properties and/or hyperparameters does GLMNET have?
modelLookup("glmnet")

# Number of possible unique hyperparameters to evaluate
possible_params <- 20

# Train the model
set.seed(4242)
model_glmnet <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "glmnet", 
                   trControl = control,
                   tuneLength = possible_params,
                   metric = "Sens")

# Output and plot
model_glmnet
plot(model_glmnet)

# Print best parameter
(best <- model_glmnet$bestTune)

# Predictions and confusion matrix
predicted <- predict(model_glmnet, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

# Feature importance
var_imp <- varImp(model_glmnet)
plot(var_imp, main = "Variable Importance with glmnet")

###################################################
# Model building - MARS                           #
###################################################

# What properties and/or hyperparameters does MARS have?
modelLookup("earth")

# Number of possible unique hyperparameters to evaluate
possible_params <- 5

# Train the model
set.seed(4242)
model_mars <- train(DEATH_EVENT ~ ., 
                   data = train, 
                   method = "earth", 
                   trControl = control,
                   tuneLength = possible_params,
                   metric = "Sens")

# Output and plot
model_mars
plot(model_mars)

# Print best parameter
(best <- model_glmnet$bestTune)

# Predictions and confusion matrix
predicted <- predict(model_mars, newdata = test)
confusionMatrix(predicted, test_outcome$DEATH_EVENT, mode = "everything")

# Feature importance
var_imp <- varImp(model_mars)
plot(var_imp, main = "Variable Importance with MARS")

###################################################
# Comparing the models                            #
###################################################

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST = model_adaboost, GLMNET = model_glmnet, 
                                 KNN = model_knn, LDA = model_lda, RF = model_rf,
                                 SVM_L = model_svm, SVM_R = model_svm_radial, 
                                 MARS = model_mars))

# Summary of the models performances
(summary_table <- summary(models_compare))

# Plot
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(models_compare, scales = scales)

# Important references
# 1. Caret Package – A Practical Guide to Machine Learning in R: https://www.machinelearningplus.com/machine-learning/caret-package/"
# 2. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5
# 3. caret package documentation 

