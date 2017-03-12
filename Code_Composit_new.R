##First set your directory
library(readxl)
library(magrittr)
library(dplyr)
library(caret)
# read_excel reads both xls and xlsx files
train <- read_excel("excavate.xlsx")
YOUR_TEST_DATA <- read_excel("Test.xlsx")
##Checking for missing pattern data
library(VIM)
mice_plot <- aggr(train, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(train), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

##Imputation of Missing data by Mice Package
#install MICE
#("mice")
##For Training Data
library(mice)
imputed_train <- mice(train, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_train)


#get complete data ( 2nd out of 5)
completeData <- complete(imputed_train,2)

target <- completeData$`I.D. of glass`
#converting into factor
target <- as.factor(target)
completeData$target <- target
completeData$inverse_RI <- 1/completeData$RI
completeData$`I.D. of glass` <- NULL
completeData$SN <- NULL
rm(target)


##Imputation for YOUR_TEST_DATA
imputed_YOUR_TEST_DATA <- mice(YOUR_TEST_DATA, m=5, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_YOUR_TEST_DATA)


#get YOUR_TEST_DATA data ( 2nd out of 5)
YOUR_TEST_DATA <- complete(imputed_YOUR_TEST_DATA,2)

target_YOUR_TEST_DATA <- YOUR_TEST_DATA$`I.D. of glass`
#converting into factor
target_YOUR_TEST_DATA <- as.factor(target_YOUR_TEST_DATA)
YOUR_TEST_DATA$target <- target_YOUR_TEST_DATA
YOUR_TEST_DATA$inverse_RI <- 1/YOUR_TEST_DATA$RI
YOUR_TEST_DATA$`I.D. of glass` <- NULL
YOUR_TEST_DATA$SN <- NULL

##Sampling[0.65:0.35]
train_index <- sample(1:nrow(completeData), nrow(completeData)*0.65)
train_new <- completeData[train_index,]
test_new <- completeData[-train_index,]

##XGBOOST Model
# now creating the model 
t1 <- Sys.time()
library(xgboost)
# creating first matrix (data and label) using train_new
dtrain<-xgb.DMatrix(data=data.matrix(train_new),label=data.matrix(train_new$target),missing=NA)
# creating second matrix (data and label) using your test data
dval<-xgb.DMatrix(data=data.matrix(test_new),label=data.matrix(test_new$target),missing=NA)
watchlist<-list(val=dval,train=dtrain)
fin_pred={}
for (eta in c(0.1,0.05,0.01) )
{
  t <- Sys.time()
  for (colsample_bytree in c(0.2,0.4,0.6))
  {
    for(subsample in c(0.4,0.8,1))
    {
      param <- list(  objective           = "multi:softmax", 
                      booster             = "gbtree",
                      eta                 = eta,
                      max_depth           =  8,
                      subsample           = subsample,
                      colsample_bytree    = colsample_bytree,
                      num_class           = 8
      )
      gc()
      set.seed(1429)
      # creating the model 
      clf <- xgb.train(   params              = param, 
                          data                = dtrain, 
                          nrounds             = 100, 
                          verbose             = 1,
                          watchlist           = watchlist,
                          maximize            = TRUE,
                          eval_metric       = "mlogloss"
      )
      
      pred_exp=predict(clf,data.matrix(YOUR_TEST_DATA),missing=NA)
      print(head(fin_pred))
      fin_pred<-cbind(fin_pred,pred_exp)
    }
  }
  print(Sys.time() - t)
}

print(Sys.time() - t1)
final_pred_YOUR_TEST_DATA <- apply(fin_pred,1,median)#Taking Median of all the models
#Accuracy 0.95-0.98
#F1 Score avg.0.96-0.98
##For YOUR_TEST_DATA
confusionMatrix<- confusionMatrix(target_YOUR_TEST_DATA,
                                  final_pred_YOUR_TEST_DATA,
                                  mode = "everything")
##Printing  accuracy and overall statistics
confusionMatrix

##For F1 score of YOUR_TEST_DATA
confusionMatrix$byClass

##Importance of features
# get the feature real names
names <-  colnames(completeData[,-10])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = clf)
head(importance_matrix)
# plot
gp = xgb.plot.importance(importance_matrix)
print(gp) 





