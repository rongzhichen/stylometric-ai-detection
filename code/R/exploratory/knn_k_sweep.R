# Exploratory (not in the paper): k sweep for a KNN classifier. Run 00_setup.R first.
###Perform crossvalidation based on different k for KNN
#initialzie the data
num_k <- c(1,2,3,4,5,6,7,8,9,10)
all_labels <- NULL
num_fold <- 10
#form the matrix of the corpus
Human_combined <- do.call(rbind, HumanCorpus$features)
GPT_combined <- do.call(rbind, GPTCorpus$features)
#combine
features <-rbind(Human_combined, GPT_combined)
#create vector to record the labels
all_labels <- c(rep(1, nrow(Human_combined)), rep(2, nrow(GPT_combined)))

set.seed(123456)  # For reproducibility
# Create stratified folds based on author labels
folds <- createFolds(all_labels, k = num_fold, list = TRUE, returnTrain = FALSE)

# Initialize lists to store data for each fold
train_data_list <- list()
train_labels_list <- list()
test_data_list <- list()
test_labels_list <- list()

# Total number of essays
total_books <- nrow(features)

# Generate training and testing sets for each fold
for (fold_idx in seq_along(folds)) {
  # Indices for the test set
  test_indices <- folds[[fold_idx]]
  
  # Indices for the training set
  train_indices <- setdiff(seq_len(total_books), test_indices)
  
  # Create training and testing data
  train_data <- features[train_indices, , drop = FALSE]
  train_labels <- all_labels[train_indices]
  
  test_data <- features[test_indices, , drop = FALSE]
  test_labels <- all_labels[test_indices]
  
  # Store the data in the lists
  train_data_list[[fold_idx]] <- train_data
  train_labels_list[[fold_idx]] <- train_labels
  test_data_list[[fold_idx]] <- test_data
  test_labels_list[[fold_idx]] <- test_labels
}
for (k in num_k){
  #initialize data to calculate the accuracy, F1
  preds_KNN <- list()
  
  converted_preds_KNN <- list()
  
  truth <- list()
  #Loop over each fold to train and evaluate your model
  for (fold_idx in 1:num_fold) {
    # Get training and testing data for the current fold
    train_data <- train_data_list[[fold_idx]]
    train_labels <- train_labels_list[[fold_idx]]
    
    test_data <- test_data_list[[fold_idx]]
    test_labels <- test_labels_list[[fold_idx]]
    
    count_1 <- sum(unlist(train_labels) == 1)
    count_2 <- sum(unlist(train_labels) == 2)
    
    
    preds_KNN[[fold_idx]] <- KNNCorpusnew(train_data, test_data, k = k)
    
    converted_preds_KNN[[fold_idx]] <- lapply(preds_KNN[[fold_idx]], function(x) {
      num_x <- as.numeric(as.character(x))  # Convert factor to numeric
      ifelse(num_x <= count_1, 1, 2)})
    
    truth[[fold_idx]] <-test_labels
  }
  #calculate the average accuracy, precision, recall, F1
  KNNResult <- list()
  
  for (rate_idx in 1:length(converted_preds_KNN)) {
    KNNOutput<- scoreOutput(truth[[rate_idx]], converted_preds_KNN[[rate_idx]])
    KNNResult[[rate_idx]] <- c(
      AccuracyKNN = KNNOutput[[1]],
      RecallKNN = KNNOutput[[2]],
      PrecisionKNN = KNNOutput[[3]],
      F1KNN = KNNOutput[[4]]
    )
  }
  
  
  KNN_matrix <- do.call(rbind, KNNResult)
  average_values_KNN <- colMeans(KNN_matrix)
  print(k)
  print(average_values_KNN)
}
