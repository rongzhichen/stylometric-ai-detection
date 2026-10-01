# SVM cost/kernel parameter sweep used to pick the SVM settings. Run 00_setup.R first.
#concatenate all the human essays together
#or explicitly treat each essay as having a separate author.
# Load the caret package
library(caret)

#initialzie the data
num_cost <- c(0.01,0.1,1,10,50,100)
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
for (cost in num_cost){
  #initialize data to calculate the accuracy, F1
  preds_SVM <- list()
  
  converted_preds_SVM <- list()
  
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
    
    
    preds_SVM[[fold_idx]] <- SVMCorpusnew(train_data, test_data,cost = cost, count_1 = count_1,count_2 = count_2)
    
    converted_preds_SVM[[fold_idx]] <- preds_SVM[[fold_idx]]
    
    truth[[fold_idx]] <-test_labels
  }
  #calculate the average accuracy, precision, recall, F1
  SVMResult <- list()
  
  for (rate_idx in 1:length(converted_preds_SVM)) {
    SVMOutput <-scoreOutput(truth[[rate_idx]], converted_preds_SVM[[rate_idx]])
    SVMOutput <- lapply(SVMOutput, function(x) if (is.nan(x)) 0 else x)
    SVMResult[[rate_idx]] <- c(
      AccuracySVM = SVMOutput[[1]],
      RecallSVM = SVMOutput[[2]],
      PrecisionSVM = SVMOutput[[3]],
      F1SVM = SVMOutput[[4]]
    )
  }
  
  SVM_matrix <- do.call(rbind, SVMResult)
  average_values_SVM <- colMeans(SVM_matrix)
  print(cost)
  print(average_values_SVM)
}
