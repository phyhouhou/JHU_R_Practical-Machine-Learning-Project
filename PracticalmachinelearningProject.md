# Practical machine learning Project: prediction on how well an exercise is performed
Yanhua Hou  

##Introduction

This project is aimed to quantify how well people perform in an exercise.
We will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is [here](http://groupware.les.inf.puc-rio.br/har). Our goal is build a model from the [training data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) to predict the manner in which they did the exercise in the [test data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

##Global settings

```r
knitr::opts_chunk$set(warning=F,message=F,cache=T,set.seed(34567),options(digits=2))
```

##Load and clean the data

```r
#download the training and test data
if(!file.exists("data")){dir.create("data")}
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url_train, destfile='./data/training.csv',method='curl')
download.file(url_test, destfile = './data/test.csv',method='curl')
#import the raw data
train_raw <- read.csv('./data/training.csv',na.strings=c('NA','','NULL'))
test_raw<-read.csv('./data/test.csv',na.strings=c('NA','','NULL'))
#check structure of the data
str(train_raw)
str(test_raw)
#load required packages 
require(caret);require(outliers);require(rpart);require(randomForest)
```

The structure of the raw training data shows that it has 19622 observations of 160 variables. The test data has 20 observations of 160 variables. Note that the data contains certain variables (such as 'X', 'user_name'...) that cannot be used as predictors and some variables are not complete and some variables are rarely variable. We need to clean the data before building models.

####Remove variables that can't be used as predictors

```r
training1<-train_raw[,-c(1:7)]
```

####Remove variables with NAs

```r
table(colMeans(is.na(training1)))
```

```
## 
##                 0 0.979308938946081 
##                53               100
```
In view of the high NAs percentage, we choose to use only features without NAs.

```r
training2<-training1[,colSums(is.na(training1))==0]
```

####Check and remove covariates with little variability

```r
index_nzv <- nearZeroVar(training2,saveMetrics=F)
ifelse(length(index_nzv)>0,training3<-training2[,-index_nzv],training3<-training2)
```

####Replace outliers in the features by medians

```r
train_vbs<-training3[,-ncol(training3)]
training4<-as.data.frame(sapply(train_vbs,function(x){rm.outlier(x,fill=T,median=T)}))
training4$classe<-training3[,ncol(training3)]
```

####Remove highly correlated variables 
Finally we find and remove highly correlated (larger than 0.9) variables to further reduce the number of relevant features. 

```r
#select out the outcome variable 'classe'
train_vbs<-training4[,-ncol(training4)]
#calculate the correlations between features
cor_Matrix<-cor(train_vbs)
highcor<-findCorrelation(cor_Matrix,cutoff=0.9)
train_tidy<-training4[,-highcor]
dim(train_tidy)
```

After the processes performed above, we get a tidy training data with 19622 observations of 48 features.

##Build the model

We first split the tidy data into a training (70%) and a test data set (30%) for cross validation.

```r
inTrain<-createDataPartition(train_tidy$classe,p=0.7,list=F)
training<-train_tidy[inTrain,]
testing<-train_tidy[-inTrain,]
```

#####Classification trees
We first try the classification tree model.

```r
modtree<-train(classe~., method="rpart",data=training)
#predicting new values
predicttree<-predict(modtree,newdata=testing)
conMtree<-confusionMatrix(predicttree,testing$classe)
conMtree$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1498  460  462  342  224
##          B   36  365   36   17  163
##          C  139  288  524  361  311
##          D    0   25    2  157   60
##          E    1    1    2   87  324
```
Note that the predictiction is rather poor.

```r
acctree<-conMtree$overall['Accuracy']
```
The accurarcy is 0.49, which is quite low.

#####Random forest
We use the random forest model below.

```r
modrf <- train(classe ~ ., data=training, method="rf", trControl=trainControl(method="cv", 3), ntree=50)
```
For the purpose of cross validation, we use the random forest model to predict new values.

```r
predictrf<-predict(modrf,newdata=testing)
conMrf<-confusionMatrix(predictrf,testing$classe)
conMrf$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1671   12    0    2    0
##          B    3 1124    4    0    0
##          C    0    3 1016    3    0
##          D    0    0    6  957    0
##          E    0    0    0    2 1082
```
Note that most variables are predicted correctly.

```r
accrf<-conMrf$overall['Accuracy']
```
The accuracy is 0.99, tremendously enhanced by the random forest algorithm. The expected out-of-sample error is 0.01. 

##Prediciton on raw test data 
In view of the high accuracy of random forest algorithm, we adopt the random forest model to predict the 20 exercise manners in the raw test data.

```r
predictrf_test<-predict(modrf,newdata=test_raw)
```
The predictions for the 20 types of the raw test data are: B, A, B, A, A, E, D, B, A, A, B, C, B, A, E, E, A, B, B, B.
