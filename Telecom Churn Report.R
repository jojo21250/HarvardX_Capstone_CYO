##### Data provided by IBM/Scott D'Angelo. You can view the GitHub page below
##### https://github.com/IBM/telco-customer-churn-on-icp4d
##### Data can also be found on my GitHub page. This is the URL used in the script.
##### https://raw.githubusercontent.com/jojo21250/HarvardX_Capstone_CYO/master/Telco-Customer-Churn.csv

#Get packages to run the script
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

#Download the data and read the CSV into an object
dl <- tempfile()
download.file("https://raw.githubusercontent.com/jojo21250/HarvardX_Capstone_CYO/master/Telco-Customer-Churn.csv", dl)
dat <- read_csv(dl)

#Remove unneeded object
rm(dl)

#Change class of character variables to factors 
dat <- dat %>% mutate_if(is.character,as.factor)

#Change class of SeniorCitizen to factor, 
#because it is just a TRUE/FALSE style observation represented by 1/0,
#and NOT a representation of some quantity
dat <- dat %>% mutate(SeniorCitizen = as.factor(SeniorCitizen))

#Check for NAs
(is.na(dat))

#Count the NAs to see if it's a significant number or not
sum(is.na(dat))

#11 NAs in a dataset of this size will not be very consequential
#Howver, every machine learning algorithm will require some na.action to run
#Use na.roughfix throughout the script, which is a common method which does not remove any columns or rows
#na.roughfix imputes the median value for missing numerics and the mode for missing factors


#Create a test index
test_index <- createDataPartition(dat$Churn, times = 1, p = 0.5, list = FALSE)

#Create training and test sets to prepare the data for machine learning
train_set <- dat[-test_index,]
test_set <- dat[test_index,]

#Set verboseIter = TRUE for trainControl so the console shows progress when fitting the various models
trainctrl <- trainControl(verboseIter = TRUE)

#mtry parameter tells RF how many splits to try at each node
mtry <- expand.grid(mtry = seq(2,10,2))

#Fit a random forest model for the data. Use all the columns except 1, which is the customerID. 
#Each customerID appears only once in the data, and this variable would not be a useful predictor.
fitRF <- train(Churn ~ ., 
               method = "rf", 
               data = train_set[,2:21], 
               na.action = na.roughfix,
               metric = "Accuracy",
               tuneGrid = mtry,
               trControl = trainctrl) 

#the high accuracy is due correct classification rate on "no", but the correct classification rate on "yes" is 50%

#Use the random forest model and create predictions for the test set
y_hat_rf <- predict(fitRF, test_set, na.action = na.roughfix)

#Inspect the confusions matrix to measure the performance of the random forest model
confusionMatrix(data = y_hat_rf,
                reference = test_set$Churn,
                positive = "Yes")
#Accuracy is 0.80, which is better than the accuracy we could achieve by simply guessing "No" every time (0.735), but not by that much

#Use varImp to check the most important variables in the random forest model
varImp(fitRF)

#k is a tuning parameter for knn which tells it how many neighbors to consider
k_tune <- expand.grid(k = seq(3,35,2))

#Try training a knn model
fitKNN <- train(Churn ~ ., 
                method = "knn", 
                data = train_set[,2:21], 
                na.action = na.roughfix,
                metric = "Accuracy",
                tuneGrid = k_tune,
                trControl = trainctrl)

#Make predictions using the knn model on the test set
y_hat_knn <- predict(fitKNN, test_set, na.action = na.roughfix)

#Inspect the confusion matrix to measure the performance of the knn model
confusionMatrix(data = y_hat_knn,
                reference = test_set$Churn,
                positive = "Yes")
#Accuracy is 0.77, which is a small downgrade compared to the random forest model. 
#There were some changes in sensitivity and specificity compared to the RF model.

#Try a glm model
fitGLM <- train(Churn ~ ., 
                method = "glm", 
                data = train_set[,2:21], 
                na.action = na.roughfix,
                metric = "Accuracy",
                trControl = trainctrl)

#Make predictions using glm
y_hat_glm <- predict(fitGLM, test_set, na.action = na.roughfix)

#Measure model performance
confusionMatrix(data = y_hat_glm,
                reference = test_set$Churn,
                positive = "Yes")
#Accuracy is 0.80, but accurate detection of "Yes" cases is up. Also, this model was fast.

#Use varImp to see which variables were important in the glm model
varImp(fitGLM)
#Again "tenure" is the most important, but there are some substantial differences as you go down the list

#Next, set up an ensemble which uses RF, KNN, and GLM together
#On the line below, define the control for the ensemble
trainctrl_ensemble <- trainControl(method = "cv",
                                   number = 5, 
                                   savePredictions = "final",
                                   classProbs = TRUE) #added this line to make CaretEnsemble work later

#tuneList can specify methods AND pass on tuning parameters for individual methods in the ensemble
#this list includes the methods and optimal tuning parameters from the models already trained above
#this will reduce the time it takes to build the ensemble
tunelist_ensemble <- list(
                          glm = caretModelSpec(method = "glm"),
                          rf = caretModelSpec(method = "rf", tuneGrid = data.frame(.mtry = 4)), #mtry value from earlier RF model
                          knn = caretModelSpec(method = "knn", tuneGrid = data.frame(.k = 29)) #k value from earlier KNN model
                          )

#use tuneList instead of methodList to pass tuning parameters
model_list <- caretList(Churn ~ .,
                        data = na.roughfix(train_set[,2:21]),
                        trControl = trainctrl_ensemble,
                        tuneList = tunelist_ensemble,
                        continue_on_fail = FALSE, 
                        preProcess = c("center","scale"))

#modelcor computes the correlation between the predictions of the different models
#Variety may make our ensemble more accurate. Using methods that are basically the same doesn't add anything.
modelCor(resamples(model_list))


ensemble <- caretEnsemble(
  model_list, 
  metric="ROC",
  trControl=trainControl(
    number=2,
    summaryFunction=twoClassSummary,
    classProbs=TRUE
  ))

#Make predictions using the ensemble
y_hat_ensemble <- predict(ensemble, test_set, na.action = na.roughfix)

#Examine the accuracy of the ensemble
confusionMatrix(data = y_hat_ensemble,
                reference = test_set$Churn,
                positive = "Yes")
#Accuracy is on par with GLM




#########################################################################
#####################VISUALIZATIONS AND OBSERVATIONS#####################
#########################################################################

#new customers are a high risk to churn, while long-standing customers are fairly unlikely
dat %>%
  ggplot(aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 2) +
  labs(x = "Tenure (months)", y = "Number of Customers") +
  ggtitle("Distribution of Customers by Tenure") +
  theme(plot.title = element_text(hjust = 0.5))

#more than half of customers with a tenure of 1 are churning 
dat %>% filter(tenure == 1) %>% summarize(mean = mean(.$Churn == "Yes")) #output is 0.62

#customers with the OnlineSecurity feature churn less
#follow-up research: find out why?
dat %>%
  ggplot(aes(x = OnlineSecurity, fill = Churn)) +
  geom_histogram(stat = "count") +
  labs(x = "Online Security", y = "Number of Customers") +
  ggtitle("Churn Rates - Online Security Subscribers") +
  theme(plot.title = element_text(hjust = 0.5))

#also true of customers with the Tech Support feature
dat %>%
  ggplot(aes(x = TechSupport, fill = Churn)) +
  geom_histogram(stat = "count") +
  labs(x = "Tech Support", y = "Number of Customers") +
  ggtitle("Churn Rates - Tech Support Subscribers") +
  theme(plot.title = element_text(hjust = 0.5))

#something intuitive: customers with contracts churn less
#impact: focus more retention efforts on month-to-month customers
dat %>%
  ggplot(aes(x = Contract, fill = Churn)) +
  geom_histogram(stat = "count") +
  labs(x = "Contract Type", y = "Number of Customers") +
  ggtitle("Churn Rates - Contract Type") +
  theme(plot.title = element_text(hjust = 0.5))

#this visualization shows that customers on the cheapest plans churn less
#interesting information from a business intelligence perspective
#could a lower price = larger revenue in the long haul?
dat %>%
  ggplot(aes(x = MonthlyCharges, fill = Churn)) +
  geom_histogram(binwidth = 6.25) +
  labs(x = "Monthly Charges", y = "Number of Customers") +
  ggtitle("Distribution of Customers by Monthly Charges") +
  theme(plot.title = element_text(hjust = 0.5))

#we should note that many (1526/3460) loyal customers with low monthly charges are
#not buying internet service. compare the output from the following statements.
#all "no internet" customers also have low bills
sum(dat$MonthlyCharges <= 70)
sum(dat$InternetService == "No")
sum(dat$InternetService == "No" & dat$MonthlyCharges <= 70)

identical(sum(dat$InternetService == "No"),
          sum(dat$InternetService == "No" & dat$MonthlyCharges <= 70))

#something that's been obvious in other charts: non-internet customers are loyal
#this might be due to their lower monthly charges
dat %>%
  ggplot(aes(x = InternetService, fill = Churn)) +
  geom_histogram(stat = "count") +
  labs(x = "Type of Internet Service", y = "Number of Customers") +
  ggtitle("Churn Rates - Internet Service Type") +
  theme(plot.title = element_text(hjust = 0.5))
