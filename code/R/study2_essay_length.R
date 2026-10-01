# Study 2: effect of essay length. Requires 200-word features, regenerate with:
#   python3 code/python/extract_function_words.py --input <essays_dir> --output data/function_words_200/<human|gpt> --max-words 200
# (see data/README.md). Run code/R/00_setup.R first.
####Test the performance by different test essay length
###########################################
#form the matrix of the corpus
Human_combined <- do.call(rbind, HumanCorpus$features)
GPT_combined <- do.call(rbind, GPTCorpus$features)
features <-rbind(Human_combined, GPT_combined)
ReduceGPTCorpus <- loadCorpus("data/function_words_200/gpt/","functionwords")
ReduceHumanCorpus <- loadCorpus("data/function_words_200/human/","functionwords")
ReduceHuman_combined <- do.call(rbind, ReduceHumanCorpus$features)
ReduceGPT_combined <- do.call(rbind, ReduceGPTCorpus$features)
#combine
reducedfeatures <-rbind(ReduceHuman_combined, ReduceGPT_combined)
#create vector to record the labels
all_labels <- c(rep(1, nrow(ReduceHuman_combined)), rep(2, nrow(ReduceGPT_combined)))
set.seed(123456)  # For reproducibility
num_fold <-10
# Create stratified folds based on author labels
folds <- createFolds(all_labels, k = num_fold, list = TRUE, returnTrain = FALSE)

# Initialize lists to store data for each fold
train_data_list <- list()
train_labels_list <- list()
test_data_list <- list()
test_labels_list <- list()

# Total number of essays
total_books <- nrow(reducedfeatures)

# Generate training and testing sets for each fold
for (fold_idx in seq_along(folds)) {
  # Indices for the test set
  test_indices <- folds[[fold_idx]]
  
  # Indices for the training set
  train_indices <- setdiff(seq_len(total_books), test_indices)
  
  # Create training and testing data
  train_data <- features[train_indices, , drop = FALSE]
  train_labels <- all_labels[train_indices]
  
  test_data <- reducedfeatures[test_indices, , drop = FALSE]
  test_labels <- all_labels[test_indices]
  
  # Store the data in the lists
  train_data_list[[fold_idx]] <- train_data
  train_labels_list[[fold_idx]] <- train_labels
  test_data_list[[fold_idx]] <- test_data
  test_labels_list[[fold_idx]] <- test_labels
}
  #initialize data to calculate the accuracy, F1
  preds_KNN <- list()
  preds_Ran <- list()
  preds_SVM <- list()
  converted_preds_KNN <- list()
  converted_preds_Ran <- list()
  converted_preds_SVM <- list()
  truth <- list()
  #Loop over each fold to train and evaluate your model
  for (fold_idx in 1:num_fold) {
    print(fold_idx)
    # Get training and testing data for the current fold
    train_data <- train_data_list[[fold_idx]]
    train_labels <- train_labels_list[[fold_idx]]
    
    test_data <- test_data_list[[fold_idx]]
    test_labels <- test_labels_list[[fold_idx]]
    
    count_1 <- sum(unlist(train_labels) == 1)
    count_2 <- sum(unlist(train_labels) == 2)
    
    
    preds_KNN[[fold_idx]] <- KNNCorpusnew(train_data, test_data, k = 1)
    preds_Ran[[fold_idx]] <- randomForestCorpusnew(train_data, test_data, count_1 = count_1,count_2 = count_2)
    preds_SVM[[fold_idx]] <- SVMCorpusnew(train_data, test_data,cost = 100,count_1 = count_1,count_2 = count_2)
    
    converted_preds_KNN[[fold_idx]] <- lapply(preds_KNN[[fold_idx]], function(x) {
      num_x <- as.numeric(as.character(x))  # Convert factor to numeric
      ifelse(num_x <= count_1, 1, 2)})
    
    converted_preds_Ran[[fold_idx]] <- preds_Ran[[fold_idx]]
    
    converted_preds_SVM[[fold_idx]] <- preds_SVM[[fold_idx]]
    
    truth[[fold_idx]] <-test_labels
  }
  #calculate the average accuracy, precision, recall, F1
  KNNResult <- list()
  RanResult <- list()
  SVMResult <- list()
  for (rate_idx in 1:length(converted_preds_KNN)) {
    KNNOutput<- scoreOutput(truth[[rate_idx]], converted_preds_KNN[[rate_idx]])
    KNNResult[[rate_idx]] <- c(
      AccuracyKNN = KNNOutput[[1]],
      RecallKNN = KNNOutput[[2]],
      PrecisionKNN = KNNOutput[[3]],
      F1KNN = KNNOutput[[4]]
    )
  }
  
  
  for (rate_idx in 1:length(converted_preds_Ran)) {
    RanOutput<- scoreOutput(truth[[rate_idx]], converted_preds_Ran[[rate_idx]])
    RanOutput <- lapply(RanOutput, function(x) if (is.nan(x)) 0 else x)
    RanResult[[rate_idx]] <- c(
      AccuracyRan = RanOutput[[1]],
      RecallRan = RanOutput[[2]],
      PrecisionRan = RanOutput[[3]],
      F1Ran = RanOutput[[4]]
    )
  }
  
  for (rate_idx in 1:length(converted_preds_SVM)) {
    SVMOutput <-scoreOutput(truth[[rate_idx]], converted_preds_SVM[[rate_idx]])
    SVMResult[[rate_idx]] <- c(
      AccuracySVM = SVMOutput[[1]],
      RecallSVM = SVMOutput[[2]],
      PrecisionSVM = SVMOutput[[3]],
      F1SVM = SVMOutput[[4]]
    )
  }
  KNN_matrix <- do.call(rbind, KNNResult)
  average_values_KNN <- colMeans(KNN_matrix)
  Ran_matrix <- do.call(rbind, RanResult)
  average_values_Ran <- colMeans(Ran_matrix)
  SVM_matrix <- do.call(rbind, SVMResult)
  average_values_SVM <- colMeans(SVM_matrix)
  print(average_values_KNN)
  print(average_values_Ran)
  print(average_values_SVM)
