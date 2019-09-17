# Install the packages in R
install.packages(c("data.table","ggplot2","plyr","dplyr","corrplot","caret","dummies","randomForest"))

# Load libraries
library(data.table)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(dummies)
library(randomForest)
library(MLmetrics)

# Set the working directory
setwd("C:/New folder/Aditi sen/Hackathon")

# Read the training data
data <- read.csv("train.csv")

# Data Pre-Processing
apply(data, 2, function(x) sum(is.na(x)))
summary(data)

data$m13 = as.factor(data$m13)
data$m1 <- ifelse(data$m1==0,0,1)
data$m2 <- ifelse(data$m2==0,0,1)
data$m3 <- ifelse(data$m3==0,0,1)
data$m4 <- ifelse(data$m4==0,0,1)
data$m5 <- ifelse(data$m5==0,0,1)
data$m6 <- ifelse(data$m6==0,0,1)
data$m7 <- ifelse(data$m7==0,0,1)
data$m8 <- ifelse(data$m8==0,0,1)
data$m9 <- ifelse(data$m9==0,0,1)
data$m10 <- ifelse(data$m10==0,0,1)
data$m11 <- ifelse(data$m11==0,0,1)
data$m12 <- ifelse(data$m12==0,0,1)

data %>% group_by(m13) %>% summarise(mean(unpaid_principal_bal), median(unpaid_principal_bal))


# Check correlation between different variables
data$m13 <- as.numeric(data$m13)

data$number_of_borrowers <- as.factor(data$number_of_borrowers)
data$insurance_type <- as.factor(data$insurance_type)
drop <- c("loan_id","source","financial_institution","origination_date","first_payment_date","loan_purpose","number_of_borrowers","insurance_type")
d = data[,!(names(data) %in% drop)]
corr_plot <- corrplot(cor(d), method = "circle", type = "upper")

d1 = data[,(names(data) %in% drop)]

tmp <- cor(d)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0


data.new <- d[,!apply(tmp,2,function(x) any(abs(x) > 0.8))]
head(data.new)

data <- cbind(d1,data.new)
loan_id <- data$loan_id
data$loan_id <- NULL
m13 <- data$m13
data$m13 <- NULL

data[,16:27] <- apply(data[,16:27], 2, function(x) as.factor(as.integer(x)))
d <- data[,c(1:7)]
d1 <- data[,c(16:27)]

d2 <- data[,c(8:15)]

# Normalizing the data
normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
}


dNorm <- as.data.frame(lapply(d2, normalize))

# Creating dummy dataframe for categorical variables

dum <- dummy.data.frame(d)

# Creating final dataset
data <- cbind(dum,dNorm,d1)
data$m13 <- as.factor(m13)
data$m13 <- as.factor(ifelse(data$m13==1,0,1))
data[,45:56] <- apply(data[,45:56], 2, function(x) as.integer(x))
# Transform "Class" to factor to perform classification and rename levels to predict class probabilities (need to be valid R variable names)
data$m13 <- as.numeric(data$m13)
data$m13 <- ifelse(data$m13==1,0,1)
data <- data[,-c(23:29)]
#######################################################################################
######################Processing the test set##########################################
#######################################################################################

# Read the test data
testdata <- read.csv("test.csv")

# Data Pre-Processing,  recoding data, normalization and creating dummies of categorical variables
apply(testdata, 2, function(x) sum(is.na(x)))
summary(testdata)

testdata$m1 <- ifelse(testdata$m1==0,0,1)
testdata$m2 <- ifelse(testdata$m2==0,0,1)
testdata$m3 <- ifelse(testdata$m3==0,0,1)
testdata$m4 <- ifelse(testdata$m4==0,0,1)
testdata$m5 <- ifelse(testdata$m5==0,0,1)
testdata$m6 <- ifelse(testdata$m6==0,0,1)
testdata$m7 <- ifelse(testdata$m7==0,0,1)
testdata$m8 <- ifelse(testdata$m8==0,0,1)
testdata$m9 <- ifelse(testdata$m9==0,0,1)
testdata$m10 <- ifelse(testdata$m10==0,0,1)
testdata$m11 <- ifelse(testdata$m11==0,0,1)
testdata$m12 <- ifelse(testdata$m12==0,0,1)


testdata$number_of_borrowers <- as.factor(testdata$number_of_borrowers)
testdata$insurance_type <- as.factor(testdata$insurance_type)
loan_id_test <- testdata$loan_id
testdata$loan_id <- NULL

d <- testdata[,c(1,2,6,7,9,12,15)]
d1 <- testdata[,c(16:27)]
d2 <- testdata[,c(3,4,5,8,10,11,13,14)]
normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
}


dNorm <- as.data.frame(lapply(d2, normalize))

dum <- dummy.data.frame(d)

# Creating final test dataset
testdata <- cbind(dum,dNorm,d1)
testdata <- testdata[,-c(23:29)]


#######################################################################################
############################Model Development##########################################
#######################################################################################

# Set random seed for reproducibility
set.seed(42)


# Create training and testing set with stratification (i.e. preserving the proportions of false/true values from the "m13" column)
train_index <- createDataPartition(data$m13, times = 1, p = 0.8, list = F)
X_train <- data[train_index,]
X_test <- data[-train_index,]
y_train <- data$m13[train_index]
y_test <- data$m13[-train_index]

## Use 10-fold cross-validation and using smote to sample data
ctrl <- trainControl(method = "cv",
                     number = 10,
                     verboseIter = T,
                     classProbs = T,
                     sampling = "smote",
                     summaryFunction = twoClassSummary,
                     savePredictions = T)

# using random forest for model development

X_train_rf <- X_train
X_train_rf$m13 <- as.factor(X_train_rf$m13)
levels(X_train_rf$m13) <- make.names(c(0, 1))
model_rf_smote <- train(m13 ~ ., data = X_train_rf, method = "rf", trControl = ctrl, verbose = T, metric = "ROC")
model_rf_smote
# prediction on validation set
preds <- predict(model_rf_smote, X_test, type = "prob")
preds$m13 <- ifelse(preds$X1>preds$X0,1,0)
preds$m13 <- as.factor(preds$m13)

Precision(y_test, preds$m13, positive = NULL)

#######################################################################################
############################Final Prediction###########################################
#######################################################################################

# prediction on test set
preds <- predict(model_rf_smote, testdata, type = "prob")
preds$m13 <- ifelse(preds$X0>preds$X1,0,1)
final <- as.data.frame(cbind(loan_id_test,preds$m13))
colnames(final) <- c("loan_id","m13")

# write the dataframe as csv file
write.csv(final,"file2r.csv",row.names = F)