#install.packages('class')
#install.packages('caret')

library(class)
library(caret)
library(randomForest)
library(e1071)

#load in a literary corpus. Filedir should be the directory of the function words, which contains one folder for
#each author. The 'featureset' argument denotes the type of features that should be used
loadCorpus <- function(filedir,featureset="functionwords",maxauthors=Inf) {
  authornames <- list.files(filedir)
  booknames <- list()
  features <- list()
  count <- 0
  
  for (i in 1:length(authornames)) {
    #print(i)
    if (count >= maxauthors) {break}
    files <- list.files(sprintf("%s%s/",filedir,authornames[i]))
    if (length(files)==0) {next}
    
    firstbook <- FALSE
    booknames[[i]] <- character()
    for (j in 1:length(files)) {
      path <- sprintf("%s%s/%s",filedir,authornames[i],files[j])
      
      fields <- strsplit(files[j],split=' --- ')[[1]]  
      
      if (sprintf("%s.txt",featureset) == fields[2]) {
        booknames[[i]] <- c(booknames[[i]], fields[1])
        count <- count+1
        M <- as.matrix(read.csv(path,sep=',',header=FALSE))  
        if (firstbook == FALSE) {
          firstbook <- TRUE
          features[[i]] <- M
        } else {
          features[[i]]  <- rbind(features[[i]],M)
        }
        
      }
    }
  }
  return(list(features=features,booknames=booknames,authornames=authornames))
}

myKNN <- function(traindata, testdata, trainlabels, k=1) {
  if (mode(traindata) == 'numeric' && !is.matrix(traindata)) {
    traindata <- matrix(traindata,nrow=1)
  }
  if (mode(testdata) == 'numeric' && !is.matrix(testdata)) {
    testdata <- matrix(testdata,nrow=1)
  }
  
  mus <- apply(traindata,2,mean) 
  sigmas <- apply(traindata,2,sd)
  for (i in 1:ncol(traindata)) {
    traindata[,i] <- (traindata[,i] - mus[i])/sigmas[i]
  }
  
  for (i in 1:ncol(testdata)) {
    testdata[,i] <- (testdata[,i]-mus[i])/sigmas[i]
  }
  traindata[is.na(traindata)] <- 0
  testdata[is.na(testdata)] <- 0
  traindata[is.infinite(traindata)] <- 0
  testdata[is.infinite(testdata)] <- 0
  preds <- knn(train = traindata, test = testdata, cl = trainlabels, k)
  return(preds)
}

discriminantCorpus <- function(traindata, testdata) {
  thetas <- NULL
  preds <- NULL
  
  #first learn the model for each author
  for (i in 1:length(traindata)) {
    words <- apply(traindata[[i]],2,sum)
    
    #some words might never occur. This will be a problem since it will mean the theta for this word is 0, which means the likelihood will be 0 if this word occurs in the training set. So, we force each word to occur at leats once
    inds <- which(words==0) 
    if (length(inds) > 0) {words[inds] <- 1}
    thetas <- rbind(thetas, words/sum(words))
  }
  
  #now classify
  for (i in 1:nrow(testdata)) {
    probs <- NULL
    for (j in 1:nrow(thetas)) {
      probs <- c(probs, dmultinom(testdata[i,],prob=thetas[j,],log=TRUE))
    }
    preds <- c(preds, which.max(probs))
  }
  return(preds)
}


KNNCorpus <- function(traindata, testdata, k = 1) {
  train <- NULL
  for (i in 1:length(traindata)) {
    train <- rbind(train, apply(traindata[[i]],2,sum))
  }
  
  for (i in 1:nrow(train)) {
    train[i,] <- train[i,]/sum(train[i,])
  }
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  trainlabels <- 1:nrow(train)
  myKNN(train, testdata, trainlabels,k)
}

randomForestCorpus <- function(traindata, testdata) {
  x <- NULL
  y <- NULL
  for (i in 1:length(traindata)) {
    x <- rbind(x,apply(traindata[[i]],2,sum))
    y <- c(y,i)
  }
  for (i in 1:nrow(x)) {
    x[i,] <- x[i,]/sum(x[i,])
  }
  
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  
  mus <- apply(x,2,mean)
  sigmas <- apply(x,2,sd)
  for (j in 1:ncol(x)) {
    x[,j] <- (x[,j] - mus[j])/sigmas[j]
    testdata[,j] <- (testdata[,j] - mus[j])/sigmas[j]
  }
  
  y <- as.factor(y)
  rf <- randomForest(x,y)
  
  preds <- numeric(nrow(testdata))
  for (i in 1:nrow(testdata)) {
    preds[i] <- predict(rf,testdata[i,])
  }
  # Plot variable importance
  return(preds)
}

SVMCorpus <- function(traindata, testdata, kernel = "linear", cost = 100) {
  # Number of authors/books in the training data
  train <- NULL
  for (i in 1:length(traindata)) {
    train <- rbind(train, apply(traindata[[i]],2,sum))
  }
  num_authors <- nrow(train)
  # Normalize the training data
  for (i in 1:num_authors) {
    words <- train[i, ]
    train[i, ] <- words / sum(words)
  }
  # Normalize the test data
  for (i in 1:nrow(testdata)) {
    testdata[i, ] <- testdata[i, ] / sum(testdata[i, ])
  }
  
  # Assign labels to training data
  trainlabels <- factor(1:num_authors)
  # Train the SVM model
  svm_model <- svm(train, trainlabels, kernel = kernel, cost = cost, scale = FALSE)
  
  # Predict on test data
  preds <- predict(svm_model, testdata)
  
  # Convert predictions to numeric if necessary
  preds_numeric <- as.numeric(as.character(preds))
  
  return(preds_numeric)
}

#Calculate Accuracy, F1, Precision, Recall
scoreOutput <-function(truth, output){
  #1 for human 2 for chatGPT
  FN <- sum(truth == 2 & output == 1)
  TP <- sum(truth == 2 & output == 2)
  TN <- sum(truth == 1 & output == 1)
  FP <- sum(truth == 1 & output == 2)
  accuracy = (TP+TN)/(TP+FP+TN+FN)
  recall = TP/(TP+FN)
  precision = TP/(TP+FP)
  F1 = 2*precision*recall/(precision+recall)
  return(list(accuracy, recall, precision, F1))
}

discriminantCorpusnew <- function(traindata, testdata) {
  thetas <- NULL
  preds <- NULL
  
  # Number of authors/books in the training data
  num_authors <- nrow(traindata)
  
  # Compute theta (word probabilities) for each author
  for (i in 1:num_authors) {
    # Get word counts for the author's book
    words <- traindata[i,]
    
    # Handle words that never occur by adding 1 (Laplace smoothing)
    inds <- which(words == 0)
    if (length(inds) > 0) {
      words[inds] <- 1
    }
    
    # Store the theta for each author
    thetas <- rbind(thetas, words/sum(words))
  }
  
  # Classify the test data
  for (i in 1:nrow(testdata)) {
    probs <- NULL
    for (j in 1:nrow(thetas)) {
    # Compute the log-likelihood of the test data under each author's model
    log_likelihood <- dmultinom(testdata[i, ], prob = thetas[j, ], log = TRUE)
    probs <- c(probs, log_likelihood)
  }
  
  # Predict the author with the highest log-likelihood
    preds <- c(preds, which.max(probs))
  }
  return(preds)
}


KNNCorpusnew <- function(traindata, testdata, k = 1) {
  # Number of authors/books in the training data
  num_authors <- nrow(traindata)
  
  # Transpose traindata to work with columns representing books
  for (i in 1:num_authors) {
    # Get word counts for each book (column)
    words <- traindata[i,]
    # Normalize word counts to get relative frequencies
    traindata[i, ] <- words / sum(words)
  }
  # Normalize the test data
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  # Assign labels to training data (each author is a unique label)
  trainlabels <- 1:nrow(traindata)
  # Call the myKNN function
  preds <- myKNN(traindata, testdata, trainlabels, k)
  
  return(preds)
}
  
randomForestCorpusnew <- function(traindata, testdata,count_1, count_2) {
  x <- traindata
  y <- c(rep(1,count_1), rep(2,count_2))
  
  for (i in 1:nrow(x)) {
    x[i,] <- x[i,]/sum(x[i,])
  }
  
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  
  mus <- apply(x,2,mean)
  sigmas <- apply(x,2,sd)
  for (j in 1:ncol(x)) {
    x[,j] <- (x[,j] - mus[j])/sigmas[j]
    testdata[,j] <- (testdata[,j] - mus[j])/sigmas[j]
  }
  
  y <- as.factor(y)
  x[is.na(x)] <- 0
  y[is.na(y)] <- 0
  x[is.infinite(x)] <- 0
  y[is.infinite(y)] <- 0
  testdata[is.na(testdata)] <- 0
  testdata[is.infinite(testdata)] <- 0
  rf <- randomForest(x,y)
  
  preds <- numeric(nrow(testdata))
  for (i in 1:nrow(testdata)) {
    preds[i] <- predict(rf,testdata[i,])
  }
  return(preds)
}  
  
randomForestCorpusnew_1 <- function(traindata, testdata,count_1, count_2) {
  x <- traindata
  y <- c(rep(1,count_1), rep(2,count_2))
  
  for (i in 1:nrow(x)) {
    x[i,] <- x[i,]/sum(x[i,])
  }
  
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  
  mus <- apply(x,2,mean)
  sigmas <- apply(x,2,sd)
  for (j in 1:ncol(x)) {
    x[,j] <- (x[,j] - mus[j])/sigmas[j]
    testdata[,j] <- (testdata[,j] - mus[j])/sigmas[j]
  }
  
  y <- as.factor(y)
  x[is.na(x)] <- 0
  #y[is.na(y)] <- 0
  x[is.infinite(x)] <- 0
  #y[is.infinite(y)] <- 0
  testdata[is.na(testdata)] <- 0
  testdata[is.infinite(testdata)] <- 0
  rf <- randomForest(x,y)
  
  preds <- numeric(nrow(testdata))
  for (i in 1:nrow(testdata)) {
    preds[i] <- predict(rf,testdata[i,])
  }
  return(preds)
} 

SVMCorpusnew <- function(traindata, testdata, kernel = "linear", cost = 100, count_1, count_2) {
  # Number of authors/books in the training data
  num_authors <- nrow(traindata)
  # Normalize the training data
  for (i in 1:num_authors) {
    words <- traindata[i, ]
    traindata[i, ] <- words / sum(words)
  }
  
  # Normalize the test data
  for (i in 1:nrow(testdata)) {
    testdata[i, ] <- testdata[i, ] / sum(testdata[i, ])
  }
  
  # Assign labels to training data (each author is a unique label)
  trainlabels <- factor(c(rep(1,count_1), rep(2,count_2)))
  # Train the SVM model
  svm_model <- svm(traindata, trainlabels, kernel = "linear", cost = cost, scale = FALSE)
  
  # Predict on test data
  preds <- predict(svm_model, testdata)
  
  # Convert predictions to numeric if necessary
  preds_numeric <- as.numeric(as.character(preds))
  
  return(preds_numeric)
}


KNNCorpus_Fun <- function(traindata, testdata, k = 1,fl) {
  # Number of authors/books in the training data
  num_authors <- nrow(traindata)
  
  # Transpose traindata to work with columns representing books
  for (i in 1:num_authors) {
    # Get word counts for each book (column)
    words <- traindata[i,]
    
    # Normalize word counts to get relative frequencies
    traindata[i, ] <- words / sum(words)
  }
  # Normalize the test data
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  # Assign labels to training data (each author is a unique label)
  trainlabels <- 1:nrow(traindata)
  # Call the myKNN function
  preds <- myKNN(traindata[,fl], testdata[,fl], trainlabels, k)
  
  return(preds)
}

discriminantCorpus_Fun <- function(traindata, testdata,fl) {
  thetas <- NULL
  preds <- NULL
  print(fl)
  # Number of authors/books in the training data
  num_authors <- nrow(traindata)
  
  # Compute theta (word probabilities) for each author
  for (i in 1:num_authors) {
    # Get word counts for the author's book
    words <- traindata[i,]
    
    # Handle words that never occur by adding 1 (Laplace smoothing)
    inds <- which(words == 0)
    if (length(inds) > 0) {
      words[inds] <- 1
    }
    
    # Store the theta for each author
    thetas <- rbind(thetas, words/sum(words))
  }
  testdata = testdata[,fl]
  thetas = thetas[,fl]
  # Classify the test data
  for (i in 1:nrow(testdata)) {
    probs <- NULL
    for (j in 1:nrow(thetas)) {
      # Compute the log-likelihood of the test data under each author's model
      log_likelihood <- dmultinom(testdata[i,], prob = thetas[j,], log = TRUE)
      probs <- c(probs, log_likelihood)
    }
    
    # Predict the author with the highest log-likelihood
    preds <- c(preds, which.max(probs))
  }
  return(preds)
}

randomForestCorpus_Fun <- function(traindata, testdata,count_1, count_2,fl) {
  x <- traindata
  y <- c(rep(1,count_1), rep(2,count_2))
  
  for (i in 1:nrow(x)) {
    x[i,] <- x[i,]/sum(x[i,])
  }
  
  for (i in 1:nrow(testdata)) {
    testdata[i,] <- testdata[i,]/sum(testdata[i,])
  }
  
  mus <- apply(x,2,mean)
  sigmas <- apply(x,2,sd)
  for (j in 1:ncol(x)) {
    x[,j] <- (x[,j] - mus[j])/sigmas[j]
    testdata[,j] <- (testdata[,j] - mus[j])/sigmas[j]
  }
  
  y <- as.factor(y)
  rf <- randomForest(x[,fl],y)
  
  preds <- numeric(nrow(testdata))
  for (i in 1:nrow(testdata)) {
    preds[i] <- predict(rf,testdata[i,fl])
  }
  return(preds)
}  

SVMCorpus_Fun <- function(traindata, testdata, kernel = "linear", cost = 100, count_1, count_2,fl) {
  # Number of authors/books in the training data
  num_authors <- nrow(traindata)
  # Normalize the training data
  for (i in 1:num_authors) {
    words <- traindata[i, ]
    traindata[i, ] <- words / sum(words)
  }
  
  # Normalize the test data
  for (i in 1:nrow(testdata)) {
    testdata[i, ] <- testdata[i, ] / sum(testdata[i, ])
  }
  
  # Assign labels to training data (each author is a unique label)
  trainlabels <- factor(c(rep(1,count_1), rep(2,count_2)))
  # Train the SVM model
  svm_model <- svm(traindata[,fl], trainlabels, kernel = "linear", cost = cost, scale = FALSE)
  
  # Predict on test data
  preds <- predict(svm_model, testdata[,fl])
  
  # Convert predictions to numeric if necessary
  preds_numeric <- as.numeric(as.character(preds))
  
  return(preds_numeric)
}

resize_matrix <- function(mat) {
  # Calculate the number of rows and columns to keep (50%)
  new_rows <- round(nrow(mat) *0.1)
  new_cols <- ncol(mat)
  # Randomly sample rows and columns
  sampled_rows <- sample(1:nrow(mat), new_rows)
  sampled_cols <- sample(1:ncol(mat), new_cols)
  # Return the reduced matrix
  return(mat[sampled_rows, sampled_cols, drop = FALSE])
}

