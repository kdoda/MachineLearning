# Include the LIBSVM package
library(e1071)

setwd('/Users/klevindoda/Desktop/BYU-I/Machine Learning/Machine Learning/08 Support Vector Machines')

########################################################Vowels###################################################
##2^-5,2^-3,2^-1,2,2^3,2^5,2^7,2^9,2^11,
consts <- c(0.03125,0.125,0.5,2,8,32,128,512,2048,8192)

##2^-15,2^-13,2^-11,2^-9,2^-7,2^-5,2^-3,2^-1,2
gammas <- c(0.00003051757,0.00012207031,0.00048828125,0.001953125,0.0078125,0.03125,0.125,0.5,2,8)

vowelDataSet <- read.csv(file = 'vowel_data.csv', head=F, sep="")

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows <- 1:nrow(vowelDataSet)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

print(testRows)

# The test set contains all the test rows
vowelTest <- vowelDataSet[testRows,]

# The training set contains all the other rows
vowelTrain <- vowelDataSet[-testRows,]

targets <- vowelTest$V14

maxAccuracy <- -1
maxGamma <- 0 
maxConst <- 0

for(c in consts)
{
  for(g in gammas)
  {
    model <- svm(V14~., data = vowelTrain, kernel = "radial", gamma = g, cost = c,type="C-classification")
    prediction <- predict(model, vowelTest[,-14])
    confusionMatrix <- table(pred = prediction, true = targets)
    accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
    if (accuracy > maxAccuracy)
    {
      maxAccuracy <- accuracy
      maxGamma <- g
      maxConst <- c
    }
  }
}

print(maxAccuracy)
print(maxGamma)
print(maxConst)

model <- svm(V14~., data = vowelTrain, kernel = "radial", gamma = maxGamma, cost = maxConst,type="C-classification")

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, vowelTest[,-14])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = targets)

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == targets
table <- prop.table(table(agreement))
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)

print(confusionMatrix)
print(table)
print(accuracy)

############################################################Letters#####################################################
lettersDataSet <- read.csv(file = 'letters.csv', head=T, sep=",")

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows <- 1:nrow(lettersDataSet)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
lettersTest <- lettersDataSet[testRows,]

# The training set contains all the other rows
lettersTrain <- lettersDataSet[-testRows,]

targets <- lettersTest$letter

maxAccuracy <- -1
maxGamma <- 0 
maxConst <- 0

##making the arrays smaller since it takes longer here
##consts <- c(0.03125,0.125,2,128,512,2048,8192)
##gammas <- c(0.00003051757,0.00012207031,0.001953125,0.125,0.5,2,8)
consts <- c(0.03125,0.125,512,2048)
gammas <- c(0.00003051757,0.00012207031,0.125,0.5,2)

for(c in consts)
{
  for(g in gammas)
  {
    model <- svm(letter~., data = lettersTrain, kernel = "radial", gamma = g, cost = c,type="C-classification")
    prediction <- predict(model, lettersTest[,-1])
    confusionMatrix <- table(pred = prediction, true = targets)
    accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
    if (accuracy > maxAccuracy)
    {
      maxAccuracy <- accuracy
      maxGamma <- g
      maxConst <- c
    }
  }
}
print(maxAccuracy)
print(maxGamma)
print(maxConst)

model <- svm(letter~., data = lettersTrain, kernel = "radial", gamma = maxGamma, cost = maxConst,type="C-classification")

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, lettersTest[,-1])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = targets)

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == targets
table <- prop.table(table(agreement))
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)

print(confusionMatrix)
print(table)
print(accuracy)

########################################################Lenses#########################################################
lensesDataSet <- read.csv(file = 'lenses.csv', head=F, sep=",")

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows <- 1:nrow(lensesDataSet)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
lensesTest <- lensesDataSet[testRows,]

# The training set contains all the other rows
lensesTrain <- lensesDataSet[-testRows,]

targets <- lensesTest$V5

maxAccuracy <- -1
maxGamma <- 0 
maxConst <- 0

##making the arrays smaller since it takes longer here
consts <- c(0.03125,0.125,2,128,512,2048,8192)
gammas <- c(0.00003051757,0.00012207031,0.001953125,0.125,0.5,2,8)

for(c in consts)
{
  for(g in gammas)
  {
    model <- svm(V5~., data = lensesTrain, kernel = "radial", gamma = g, cost = c,type="C-classification")
    prediction <- predict(model, lensesTest[,-5])
    confusionMatrix <- table(pred = prediction, true = targets)
    accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
    if (accuracy > maxAccuracy)
    {
      maxAccuracy <- accuracy
      maxGamma <- g
      maxConst <- c
    }
  }
}
print(maxAccuracy)
print(maxGamma)
print(maxConst)

model <- svm(V5~., data = lensesTrain, kernel = "radial", gamma = maxGamma, cost = maxConst,type="C-classification")

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, lensesTest[,-5])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = targets)

# Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == targets
table <- prop.table(table(agreement))
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)

print(confusionMatrix)
print(table)
print(accuracy)
