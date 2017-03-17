# HR-Analytics
Decision Tree model to predict if an employee will leave the company or not
#Install all the library packages
install.packages("caret",dependencies = TRUE)
install.packages("plyr")
install.packages("dplyr")
installed.packages("C50")
installed.packages("kernlab")
installed.packages("rpart")
installed.packages("rpart.plot")


library(rpart.plot)
library(rpart)
library(plyr)
library(caret)
library(dplyr)
library(C50)
library(kernlab)

#Load the data
HR_full <- read.csv("C:/Users/Taaniya Dhar/Desktop/HR_analytics_dataset.csv")

#View Data
View(HR_full)
str(HR_full)
class(HR_full)
summary(HR_full)
table(HR_full$left)

#Data Cleaning
# Is there missing data?
sum(is.na(HR_full))
# Remove missing data rows
is.na(HR_full)
CleanHRData <- na.omit(HR_full)
CleanHRData <- droplevels(CleanHRData)
summary(CleanHRData$left)
head(CleanHRData)

# Zero variance

nzv <- nearZeroVar(CleanHRData)
CleanHRData <- CleanHRData[,-nzv]

# Lets check correlations , see - nomvars to exclude nominal variables
nomvars <- c(9:10)
corrMatrix <- cor(CleanHRData[,-nomvars], use = "pairwise.complete.obs")
corrMatrix

#Heatmap for correlation matrix
library(reshape2)
melted_corrMatrix <- melt(corrMatrix)
head(melted_corrMatrix)
library(ggplot2)
ggplot(data = melted_corrMatrix, aes(x=Var1, y=Var2, fill=value))+
geom_tile()

#Exploratory Data Analysis done on the cleaned dataset for employees who have left the company
hr_hist <- CleanHRData %>% filter(left==1)
par(mfrow=c(1,3))
hist(hr_hist$satisfaction_level,col="#3090C7", main = "Satisfaction level") 
hist(hr_hist$last_evaluation,col="#3090C7", main = "Last evaluation")
hist(hr_hist$average_montly_hours,col="#3090C7", main = "Average montly hours")

par(mfrow=c(1,2))
hist(hr_hist$Work_accident,col="#3090C7", main = "Work accident")
plot(hr_hist$salary,col="#3090C7", main = "Salary")


#Converting variables to categorical variables
CleanHRData$satisfaction_level <- cut(CleanHRData$satisfaction_level,c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),labels=c("0.00-0.09","0.10-0.91","0.20-0.29","0.30-0.39","0.40-0.49","0.50-0.59","0.60-0.69","0.70-0.79","0.80-0.89","0.90-0.10"))
CleanHRData$last_evaluation <- cut(CleanHRData$last_evaluation,c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),labels=c("0.00-0.09","0.10-0.91","0.20-0.29","0.30-0.39","0.40-0.49","0.50-0.59","0.60-0.69","0.70-0.79","0.80-0.89","0.90-0.10"))
CleanHRData$average_montly_hours <- cut(CleanHRData$average_montly_hours,c(95,120,145,170,195,220,245,270,295,320),labels = c("95-119","120-144","145-169","170-194","195-219","220-244","245-269","270-294","295-320"))

#Converting Variables to numeric and factors for running classification models
CleanHRData[, 1:10] <-lapply(CleanHRData[,1:10], as.numeric)
nomvars <- c(1,2,4,7,9:10)
CleanHRData[,nomvars] <-lapply(CleanHRData[,nomvars], as.factor)
str(HR_full)


#setting the seed
set.seed(11111)



# Lets do stratified sampling. Select rows to based on Class variable as strata
TrainingDataIndex <- createDataPartition(CleanHRData$left, p=0.70, list = FALSE)
TrainingDataIndex
# Create Training Data as subset of Bank dataset with row index numbers as identified above and all columns
TrainData <- CleanHRData[TrainingDataIndex,]
head(TrainData)
# Everything else not in training is test data. Note the - (minus)sign
TestData <- CleanHRData[-TrainingDataIndex,]
head(TestData)
table(TrainData$left)
table(TestData$left)

# We will use 10 fold cross validation to train and evaluate model
TrainingParameters <- trainControl(method = "repeatedcv", number = 10)


#Decision Classification Model
# Train a model with above parameters. We will use C5.0 algorithm
DecTreeModel <- train(left ~ ., data = TrainData, 
                      method = "C5.0",
                      trControl= TrainingParameters,
                      na.action = na.omit)

# check tree
DecTreeModel
# Plot performance
plot.train(DecTreeModel)
ggplot(DecTreeModel)

#PREDICTIONS
DTPredictions <-predict(DecTreeModel, TestData, na.action = na.pass)
#see pridictions
DTPredictions

#Confusion Matrix
cmDT <-confusionMatrix(DTPredictions, TestData$left, positive="1")
cmDT$overall
cmDT$byClass


#Creating a decision tree
n <- nrow(CleanHRData)
idx <- sample(n, n * .66)

# Make a few modications
CleanHRData %>% 
  mutate(left = factor(left, labels = c("Remain", "Left")),
    salary = ordered(salary, c("low", "medium", "high"))
  ) -> 
  d

train <- d[idx, ]
test <- d[-idx, ]
tree <- rpart(left ~ ., data = train)
res <- predict(tree, test)

#Plot the decision tree
rpart.plot(tree, type = 2, fallen.leaves = F, cex = NULL, extra = 3)
