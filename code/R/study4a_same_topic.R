# Study 4, scenario 1: train and test on the same topic (paper Sec. 5.4.1). Run 00_setup.R first.
#Using LOOCV to test the accuracy for only use the same topic as training data
average_values_KNN <- list()
average_values_DIS <- list()
average_values_Ran <- list()
average_values_SVM <- list()
for (topic in 1:length(HumanCorpus$authornames)){
  print(topic)
  humanfeatures <- HumanCorpus$features[[topic]] #select the essays on this particular topic 
  GPTfeatures <- GPTCorpus$features[[topic]]
  features <- rbind(humanfeatures, GPTfeatures) #this is a matrix of both human and GPT essays

  numHuman<- nrow(humanfeatures)
  testdata <- NULL
  traindata <- NULL
  truth <- NULL
  Dispredictions <- NULL
  KNNpredictions <- NULL
  Ranpredictions <- NULL
  SVMpredictions <- NULL
  for (j in 1:nrow(features)) {
    if (j<= numHuman){
      temp<-numHuman -1
      truth <- c(truth, 1)
    }
    if (j > numHuman){
      temp<-numHuman
      truth <- c(truth, 2)}
    testdata <- matrix(features[j,,drop = FALSE],nrow=1)
    traindata <- features[-j,,drop = FALSE]
    
    count_1 <- temp
    count_2 <- nrow(features) - temp -1
    Dispred <- discriminantCorpusnew(traindata, testdata) 
    if (Dispred<=temp){Dispre <- 1}
    else{Dispre<-2}
    Dispredictions <- c(Dispredictions, Dispre)
    
    KNNpred <- as.numeric(KNNCorpusnew(traindata, testdata, k = 1))
    if (KNNpred<=temp){KNNpre <- 1}
    else{KNNpre<-2}
    KNNpredictions <- c(KNNpredictions, KNNpre)
    
    Ranpred <- randomForestCorpusnew(traindata, testdata, count_1 = count_1,count_2 = count_2)
    Ranpredictions <- c(Ranpredictions, Ranpred)
    
    SVMpred <- SVMCorpusnew(traindata, testdata,cost = 100, count_1 = count_1,count_2 = count_2)
    SVMpredictions <- c(SVMpredictions, SVMpred)}
  
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
write_xlsx(my_data, "results/tables/KNN_onlytopic.xlsx")

my_data <- do.call(rbind, average_values_DIS)
colnames(my_data) <- c("Accuracy", "Recall", "Precision", "F1")
# Convert to data frame (ensures correct format)
my_data <- as.data.frame(my_data)
write_xlsx(my_data, "results/tables/DIS_onlytopic.xlsx")

my_data <- do.call(rbind, average_values_Ran)
colnames(my_data) <- c("Accuracy", "Recall", "Precision", "F1")
# Convert to data frame (ensures correct format)
my_data <- as.data.frame(my_data)
write_xlsx(my_data, "results/tables/Ran_onlytopic.xlsx")

my_data <- do.call(rbind, average_values_SVM)
colnames(my_data) <- c("Accuracy", "Recall", "Precision", "F1")
# Convert to data frame (ensures correct format)
my_data <- as.data.frame(my_data)
write_xlsx(my_data, "results/tables/SVM_onlytopic.xlsx")