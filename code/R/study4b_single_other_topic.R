# Study 4, scenario 2: train on a single unrelated topic (paper Sec. 5.4.2). Run 00_setup.R first.
#Using LOOCV to test the accuracy for only use the other one topic as training data
average_values_KNN <- list()
average_values_DIS <- list()
average_values_Ran <- list()
average_values_SVM <- list()
for (topic in 1:length(HumanCorpus$authornames)){
  print(topic)
  #use the next topic as training data. For the last one, use the first topic as training data.
  if (topic == length(HumanCorpus$authornames)){
    temp <- 1
  }
  else{
    temp <- topic+1
  }
  humanfeatures <- HumanCorpus$features[[topic]] #select the essays on this particular topic 
  GPTfeatures <- GPTCorpus$features[[topic]]
  features <- rbind(humanfeatures, GPTfeatures) #this is a matrix of both human and GPT essays
  humanfeatures_1 <- HumanCorpus$features[[temp]] #select the essays on this particular topic 
  GPTfeatures_1 <- GPTCorpus$features[[temp]]
  features_1 <- rbind(humanfeatures_1, GPTfeatures_1) #this is a matrix of both human and GPT essays
  
  
  numHuman<- nrow(humanfeatures)
  testdata <- NULL
  traindata <- NULL
  truth <- c(rep(1, nrow(humanfeatures)), rep(2, nrow(GPTfeatures)))
  Dispredictions <- NULL
  KNNpredictions <- NULL
  Ranpredictions <- NULL
  SVMpredictions <- NULL
    testdata <- features
    traindata <- features_1
    
    count_1 <- nrow(humanfeatures_1)
    count_2 <- nrow(GPTfeatures_1)
    
    Dispred <- discriminantCorpusnew(traindata, testdata) 
    Dispredictions <- lapply(Dispred, function(x) {
      num_x <- as.numeric(as.character(x))  # Convert factor to numeric
      ifelse(num_x <= numHuman, 1, 2)})
    
    KNNpred <- as.numeric(KNNCorpusnew(traindata, testdata, k = 1))
    KNNpredictions <- lapply(KNNpred, function(x) {
      num_x <- as.numeric(as.character(x))  # Convert factor to numeric
      ifelse(num_x <= numHuman, 1, 2)})
    
    Ranpredictions <- randomForestCorpusnew_1(traindata, testdata, count_1 = count_1,count_2 = count_2)
    
    SVMpredictions <- SVMCorpusnew(traindata, testdata,cost = 100, count_1 = count_1,count_2 = count_2)
    
  
    #calculate the average accuracy, precision, recall, F1
    DISResult <- list()
    KNNResult <- list()
    RanResult <- list()
    SVMResult <- list()
    KNNOutput<- scoreOutput(truth, KNNpredictions)
    KNNOutput <- lapply(KNNOutput, function(x) if (is.nan(x)) 0 else x)
    KNNResult <- c(
      AccuracyKNN = KNNOutput[[1]],
      RecallKNN = KNNOutput[[2]],
      PrecisionKNN = KNNOutput[[3]],
      F1KNN = KNNOutput[[4]]
    )
    
    DISOutput <-scoreOutput(truth, Dispredictions)
    DISOutput <- lapply(DISOutput, function(x) if (is.nan(x)) 0 else x)
    DISResult <- c(
      AccuracyDIS = DISOutput[[1]],
      RecallDIS = DISOutput[[2]],
      PrecisionDIS = DISOutput[[3]],
      F1DIS = DISOutput[[4]]
    )
    RanOutput<- scoreOutput(truth, Ranpredictions)
    RanOutput <- lapply(RanOutput, function(x) if (is.nan(x)) 0 else x)
    RanResult <- c(
      AccuracyRan = RanOutput[[1]],
      RecallRan = RanOutput[[2]],
      PrecisionRan = RanOutput[[3]],
      F1Ran = RanOutput[[4]]
    )
    
    
    SVMOutput <-scoreOutput(truth, SVMpredictions)
    SVMOutput <- lapply(SVMOutput, function(x) if (is.nan(x)) 0 else x)
    SVMResult <- c(
      AccuracySVM = SVMOutput[[1]],
      RecallSVM = SVMOutput[[2]],
      PrecisionSVM = SVMOutput[[3]],
      F1SVM = SVMOutput[[4]]
    )
    average_values_KNN[[topic]] <- KNNResult
    average_values_DIS[[topic]] <- DISResult
    average_values_Ran[[topic]] <- RanResult
    average_values_SVM[[topic]] <- SVMResult
    
}
#export data
my_data <- do.call(rbind, average_values_KNN)
colnames(my_data) <- c("Accuracy", "Recall", "Precision", "F1")
# Convert to data frame (ensures correct format)
my_data <- as.data.frame(my_data)
write_xlsx(my_data, "results/tables/KNN_othertopic.xlsx")

my_data <- do.call(rbind, average_values_DIS)
colnames(my_data) <- c("Accuracy", "Recall", "Precision", "F1")
# Convert to data frame (ensures correct format)
my_data <- as.data.frame(my_data)
write_xlsx(my_data, "results/tables/DIS_othertopic.xlsx")

my_data <- do.call(rbind, average_values_Ran)
colnames(my_data) <- c("Accuracy", "Recall", "Precision", "F1")
# Convert to data frame (ensures correct format)
my_data <- as.data.frame(my_data)
write_xlsx(my_data, "results/tables/Ran_othertopic.xlsx")

my_data <- do.call(rbind, average_values_SVM)
colnames(my_data) <- c("Accuracy", "Recall", "Precision", "F1")
# Convert to data frame (ensures correct format)
my_data <- as.data.frame(my_data)
write_xlsx(my_data, "results/tables/SVM_othertopic.xlsx")