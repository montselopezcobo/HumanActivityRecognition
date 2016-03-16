######################################################################
########                MACHINE LEARNING          ##########
######################################################################


#####  COURSE PROJECT 1  ######
#------------------------------

setwd("~/coursera/00_Assignments/08_Machine_Learning")
if (!file.exists("./CourseProject1_2016")) {dir.create("./CourseProject1_2016")}
setwd("./CourseProject1_2016")
if (!file.exists("./Data")) {dir.create("./Data")}
if (!file.exists("./Output")) {dir.create("./Output")}


#--- Libraries ---------
require(rpart)
require(randomForest)
require(dplyr)


#--- Download the files -----------
if (!file.exists("./Data/pml-training.csv")) {
      fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
      download.file(fileUrl, destfile = "./Data/pml-training.csv", method = "curl")
      dateDownloaded <- date()
      save(dateDownloaded, file = "./Data/dateDownloaded_train.txt")
}

if (!file.exists("./Data/pml-testing.csv")) {
      fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
      download.file(fileUrl, destfile = "./Data/pml-testing.csv", method = "curl")
      dateDownloaded <- date()
      save(dateDownloaded, file = "./Data/dateDownloaded_test.txt")
}

#--- Reading files ------------
if (!file.exists("./Data/train_original.RData")) {
train_original <- read.csv("./Data/pml-training.csv", na.strings = c("NA", "#DIV/0!"))
save("train_original", file="./Data/train_original.RData")
} else {
      load("./Data/train_original.RData")     
}
if (!file.exists("./Data/test_original.RData")) {
test_original <- read.csv("./Data/pml-testing.csv", na.strings = c("NA", "#DIV/0!"))
save("test_original", file="./Data/test_original.RData")
} else {
      load("./Data/test_original.RData")     
}


#--- INCLUDE ALL CASES (DELETING DERIVED VBLES) -------
trainALL <- train_original[ ,!colSums(is.na(train_original)) >= sum(train_original$new_window == "no")]
max(colSums(is.na(trainALL)))
# [1] 0 # All NA have been removed

#--- Delete ID variables: from X to new_window
#    (When trying a TREE algorith, the only classification variable was X)

trainALL <- trainALL[,-c(1:7)]

#--- SPLIT TRAIN_ORIGINAL INTO TRAIN AND TEST (FOR VALIDATION)
set.seed(200)
inTrain = createDataPartition(trainALL$classe, p = 0.7)[[1]]
train = trainALL[inTrain,]
valid = trainALL[-inTrain,]

train$classe <- as.factor(train$classe)
valid$classe <- as.factor(valid$classe)


#---              TRAINING MODELS
#=======================================================


#--- 1. TREE
#-------------------
set.seed(999)
if (!file.exists("./Output/modelTREE1.rds")) {
      modelTREE1 <- rpart(classe~., train)
      saveRDS(modelTREE1, "./Output/modelTREE1.rds")
} else {
      readRDS("./Output/modelTREE1.rds")
}
modelTREE1

modelTREE1$variable.importance
(ConfMatTREE1 <-as.matrix(table(train$classe, predict(modelTREE1, train,type="vector"))))
#In Sample error (training set)
(ISerrorRF1 <- 1 - sum(diag(ConfMatTREE1))/sum(ConfMatTREE1))  

  #      1    2    3    4    5
  # A 3593  108   87   44   74
  # B  505 1558  243  175  177
  # C   54  114 1933  151  144
  # D  212   64  351 1400  225
  # E   93  168  306  131 1827
# 
# [1] 0.2493994



#--- 2. RANDOM FOREST
#-------------------------
set.seed(999)
if (!file.exists("./Output/modelRF1.rds")) {
      modelRF1 <- randomForest(classe ~., train, importance=TRUE)
      saveRDS(modelRF1, "./Output/modelRF1.rds")
} else {
      modelRF1 <- readRDS("./Output/modelRF1.rds")     
}
modelRF1
 
# Call:
#  randomForest(formula = classe ~ ., data = train, importance = TRUE) 
#                Type of random forest: classification
#                      Number of trees: 500
# No. of variables tried at each split: 7
# 
#         OOB estimate of  error rate: 0.48%
# Confusion matrix:
#      A    B    C    D    E class.error
# A 3900    4    1    0    1 0.001536098
# B   14 2638    6    0    0 0.007524454
# C    0   10 2385    1    0 0.004590985
# D    0    0   22 2228    2 0.010657194
# E    0    0    0    5 2520 0.001980198



#--- Prediction on validation set
ConfMatRF1 <- confusionMatrix(valid$classe, predict(modelRF1, valid))
OOSerrorRF1 <- 1 - ConfMatRF1$overall[[1]]

# Artesanal:
# predictionConfMatRF1 <-as.matrix(table(valid$classe, predict(modelRF1, valid)))
# OOSerrorRF1 <- 1-sum(diag(predictionConfMatRF1))/sum(predictionConfMatRF1)
 
  #      A    B    C    D    E
  # A 1672    1    1    0    0
  # B    5 1128    6    0    0
  # C    0    6 1018    2    0
  # D    0    0    7  956    1
  # E    0    0    0    8 1074
# 
# [1] 0.006457094




# TUNING OF MODEL
#--------------------

#--- Select features
varImpPlot(modelRF1, n.var=12, main="Variable importance for modelRF1")
importance <- as.data.frame(modelRF1$importance)
selectedFeatures <- rownames(importance[order(importance$MeanDecreaseAccuracy, 
                                              decreasing = TRUE),])[1:12]

if (!file.exists("./Output/modelRF1b.rds")) {
      modelRF1b <- randomForest(classe ~., train[,c("classe", selectedFeatures)], 
                                importance=TRUE)
      saveRDS(modelRF1b, "./Output/modelRF1b.rds")
} else {
      modelRF1b <- readRDS("./Output/modelRF1b.rds")     
}
modelRF1b
 
# Call:
#  randomForest(formula = classe ~ ., data = train[, c("classe",      selectedFeatures)], importance = TRUE) 
#                Type of random forest: classification
#                      Number of trees: 500
# No. of variables tried at each split: 3
# 
#         OOB estimate of  error rate: 1.06%
# Confusion matrix:
#      A    B    C    D    E class.error
# A 3892   13    1    0    0 0.003584229
# B   20 2598   30   10    0 0.022573363
# C    1   14 2374    7    0 0.009181970
# D    0    3   21 2226    2 0.011545293
# E    0    8    6    9 2502 0.009108911


#--- Prediction on validation set
ConfMatRF1b <- confusionMatrix(valid$classe, predict(modelRF1b, valid))
OOSerrorRF1b <- 1 - ConfMatRF1b$overall[[1]]

# Artesanalmente:
# predictionConfMatRF1b <-as.matrix(table(valid$classe, predict(modelRF1b, valid)))
# OOSerrorRF1 <- 1-sum(diag(predictionConfMatRF1))/sum(predictionConfMatRF1)
   
  #      A    B    C    D    E
  # A 1666    6    2    0    0
  # B    9 1107   15    8    0
  # C    0    4 1015    7    0
  # D    0    1    6  957    0
  # E    0    3    5    6 1068
# 
# [1] 01223449



#--- PASSING THE MODEL TO THE TEST SET
#----------------------------------------

# PREPROCESS TEST SET
#---------------------
#--- RANDOM FOREST: Prediction on validation set
predictionTEST <- as.character(predict(modelRF1, test_original))



















