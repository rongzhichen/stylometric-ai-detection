# Exploratory (not in the paper): which individual function words separate the classes. Run 00_setup.R first.
#Are there a small number (e.g. one or two) of function words which are helping you identify ChatGPT text, 
#or are your results being driven by small variations in all the function words?

# Initialize vectors to store p-values and mean differences
#form the matrix of the corpus
Human_combined <- do.call(rbind, HumanCorpus$features)
GPT_combined <- do.call(rbind, GPTCorpus$features)
#combine
data <-list(Human_combined, GPT_combined)
p_values <- numeric(ncol(data[[1]]))
mean_diff <- numeric(ncol(data[[1]]))
for (i in 1:length(data)){
  for (j in 1:nrow(data[[i]])){
      words <- data[[i]][j,]
      inds <- which(words==0) 
      if (length(inds) > 0) {words[inds] <- 1}
      data[[i]][j, ] <-  words/sum(words)
    }
  }

# Loop over each function word
for (i in 1:ncol(data[[1]])) {
  # Split data by class
  class1_data <- data[[1]][,i]
  class2_data <- data[[2]][,i]
  
  # Perform t-test
  t_test_result <- t.test(class1_data, class2_data)
  
  # Store p-value and mean difference
  p_values[i] <- t_test_result$p.value
  mean_diff[i] <- abs(mean(class1_data) - mean(class2_data))
}
p_values
# Adjust p-values for multiple comparisons (e.g., using Bonferroni correction)
p_adjust <- p.adjust(p_values, method = "bonferroni")
which(p_adjust == 0)

##################################
#Determine the accuracy, recall, precision and F1 using partial functin words
Human_combined <- do.call(rbind, HumanCorpus$features)
GPT_combined <- do.call(rbind, GPTCorpus$features)
features <-rbind(Human_combined, GPT_combined)
#create vector to record the labels
all_labels <- c(rep(1, nrow(Human_combined)), rep(2, nrow(GPT_combined)))
fl = c(9,45,60)
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
#initialize data to calculate the accuracy, F1
preds_KNN <- list()
preds_DIS <- list()
preds_Ran <- list()
preds_SVM <- list()
converted_preds_KNN <- list()
converted_preds_DIS <- list()
converted_preds_Ran <- list()
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
  
  preds_KNN[[fold_idx]] <- KNNCorpus_Fun(train_data, test_data, k = 1,fl)
  preds_DIS[[fold_idx]] <- discriminantCorpus_Fun(train_data, test_data,fl)
  preds_Ran[[fold_idx]] <- randomForestCorpus_Fun(train_data, test_data, count_1 = count_1,count_2 = count_2,fl=fl)
  preds_SVM[[fold_idx]] <- SVMCorpus_Fun(train_data, test_data,kernel = "linear", cost = 100,count_1 = count_1,count_2 = count_2,fl)
  
  converted_preds_KNN[[fold_idx]] <- lapply(preds_KNN[[fold_idx]], function(x) {
    num_x <- as.numeric(as.character(x))  # Convert factor to numeric
    ifelse(num_x <= count_1, 1, 2)})
  converted_preds_DIS[[fold_idx]] <- lapply(preds_DIS[[fold_idx]], function(x) {
    num_x <- as.numeric(as.character(x))  # Convert factor to numeric
    ifelse(num_x <= count_1, 1, 2)})
  
  converted_preds_Ran[[fold_idx]] <- preds_Ran[[fold_idx]]
  
  converted_preds_SVM[[fold_idx]] <- preds_SVM[[fold_idx]]
  
  truth[[fold_idx]] <-test_labels
}
#calculate the average accuracy, precision, recall, F1
DISResult <- list()
KNNResult <- list()
RanResult <- list()
SVMResult <- list()
for (rate_idx in 1:length(converted_preds_KNN)) {
  KNNOutput<- scoreOutput(truth[[rate_idx]], converted_preds_KNN[[rate_idx]])
  KNNOutput <- lapply(KNNOutput, function(x) if (is.nan(x)) 0 else x)
  KNNResult[[rate_idx]] <- c(
    AccuracyKNN = KNNOutput[[1]],
    RecallKNN = KNNOutput[[2]],
    PrecisionKNN = KNNOutput[[3]],
    F1KNN = KNNOutput[[4]]
  )
}

for (rate_idx in 1:length(converted_preds_DIS)) {
  DISOutput <-scoreOutput(truth[[rate_idx]], converted_preds_DIS[[rate_idx]])
  DISOutput <- lapply(DISOutput, function(x) if (is.nan(x)) 0 else x)
  DISResult[[rate_idx]] <- c(
    AccuracyDIS = DISOutput[[1]],
    RecallDIS = DISOutput[[2]],
    PrecisionDIS = DISOutput[[3]],
    F1DIS = DISOutput[[4]]
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
  SVMOutput <- lapply(SVMOutput, function(x) if (is.nan(x)) 0 else x)
  SVMResult[[rate_idx]] <- c(
    AccuracySVM = SVMOutput[[1]],
    RecallSVM = SVMOutput[[2]],
    PrecisionSVM = SVMOutput[[3]],
    F1SVM = SVMOutput[[4]]
  )
}

KNN_matrix <- do.call(rbind, KNNResult)
average_values_KNN <- colMeans(KNN_matrix)
DIS_matrix <- do.call(rbind, DISResult)
average_values_DIS <- colMeans(DIS_matrix)
Ran_matrix <- do.call(rbind, RanResult)
average_values_Ran <- colMeans(Ran_matrix)
SVM_matrix <- do.call(rbind, SVMResult)
average_values_SVM <- colMeans(SVM_matrix)
print(average_values_KNN)
print(average_values_DIS)
print(average_values_Ran)
print(average_values_SVM)