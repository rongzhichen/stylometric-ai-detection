# Figure 1: multidimensional scaling of the function-word vectors (paper Sec. 3.3). Run 00_setup.R first.
#Visualization MDS
#Similarity of Topic

# Initialize an empty matrix
x <- NULL
# Loop through each feature set in GPTCorpus$features
for (i in 1:length(GPTCorpus$features)) {
  # Sum the features column-wise and bind them to the matrix x
  x <- rbind(x, apply(GPTCorpus$features[[i]], 2, sum))
}
# Loop through each feature set in HumanCorpus$features
for (i in 1:length(HumanCorpus$features)) {
  # Sum the features column-wise and bind them to the matrix y
  x <- rbind(x, apply(HumanCorpus$features[[i]], 2, sum))
}

# Normalize each row in x so that the sum of each row equals 1
for (i in 1:nrow(x)) {
  x[i,] <- x[i,] / sum(x[i,])
}
# Standardize each column in x
for (j in 1:ncol(x)) {
  x[,j] <- (x[,j] - mean(x[,j])) / sd(x[,j])
}
colors <- c(rep("red",length(GPTCorpus$features) ), rep("blue", length(HumanCorpus$features)))
# Calculate the distance matrix and perform classical MDS
d1 <- dist(x)
pts_1 <- cmdscale(d1)
# Plot
plot(pts_1, type = "n", main="Subject Similarity Visualization",
     xlab="Dimension 1", ylab="Dimension 2", 
     xlim=c(min(pts_1[,1]) -1, max(pts_1[,1]) + 1),
     ylim=c(min(pts_1[,2]) -1, max(pts_1[,2]) + 1),
     col = colors)
# Add the first 100 points with red labels
text(pts_1[1:length(GPTCorpus$features),1], pts_1[1:length(GPTCorpus$features),2], labels = 1:length(GPTCorpus$features), col = "red", cex = 0.7)

# Add the last 100 points with blue labels
text(pts_1[(length(GPTCorpus$features)+1):nrow(x),1], pts_1[(length(GPTCorpus$features)+1):nrow(x),2], labels = 1:length(HumanCorpus$features), col = "blue", cex = 0.7)

#Add legend
legend("topright", legend = c("GPT", "Human"), col=c("red", "blue"),pch=20,cex = 0.8)
# Add grid lines
grid()


##############################################################
#plot each essay
n_G = 0
n_H = 0
x <- NULL
for (i in 1:length(GPTCorpus$features)){
  for (j in 1:nrow(GPTCorpus$features[[i]])){
    x <- rbind(x,GPTCorpus$features[[i]][j,])
    n_G = n_G +1
  }
}
for (i in 1:length(HumanCorpus$features)){
  for (j in 1:nrow(HumanCorpus$features[[i]])){
    x <- rbind(x,HumanCorpus$features[[i]][j,])
    n_H = n_H +1
  }
}
for (i in 1:nrow(x)) { x[i,] <- x[i,] / sum(x[i,]) }

for (j in 1:ncol(x)) { x[,j] <- (x[,j] - mean(x[,j]))/sd(x[,j]) }

colors_2 <- c(rep("darkkhaki",n_G ), rep("skyblue3", n_H))
d_2 <- dist(x)
pts_2 <- cmdscale(d_2)
# Plot
plot(pts_2, main = "Visualizing Similarities Among All the Essays Written by GPT",
     xlab = "Dimension 1", ylab = "Dimension 2",
     xlim = range(pts_2[, 1]) + c(-1, 1), 
     ylim = range(pts_2[, 2]) + c(-1, 1),
     col = colors_2,
     pch = 20)
# Add grid lines for better readability
#Add legend
legend("topright", legend = c("GPT", "Human"), col=c("darkkhaki", "skyblue3"),pch=20,cex = 0.8)
grid()