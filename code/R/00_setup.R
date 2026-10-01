# Run from the repository root (e.g. open stylometric-ai-detection.Rproj or setwd() to it).
# Loads packages, the stylometry helper functions, and the two function-word corpora.
# Each essay becomes one 71-dimensional count vector (70 function words + non-function words).
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)
#Load required package
library(caret)
library(writexl)
library(class)
library(randomForest)
library(e1071)
#Set Up
source("code/R/stylometryfunctions.R")
source("code/R/reducewords.R")
GPTCorpus <- loadCorpus("data/function_words/gpt/","functionwords")
HumanCorpus <- loadCorpus("data/function_words/human/","functionwords")
