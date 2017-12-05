library(ggplot2)
library(reshape2)
setwd("~/Documents/U4/Comp401/output")

predictions <- read.csv("originalPredictions.csv", header = TRUE)
unswitched <- rep(0, length(predictions) - 3)
classifier.name <- colnames(predictions[4:length(predictions)])
actual.stage <- as.vector(predictions[, "Actual"])
n <- length(actual.stage)

for(i in 4:length(predictions)){
    v <- as.vector(predictions[,i])
    unswitched[i-3] <- round(100*length(which(actual.stage == v))/n, 3)
}


### RESULTS WHEN TRAINING DATA AND TESTING DATA ARE SWITCHED
switchedPredictions <- read.csv("switchedPredictions.csv", header = TRUE)
switched <- rep(0, length(switchedPredictions) - 3)
classifier.name.s <- colnames(switchedPredictions[4:length(switchedPredictions)])
actual.stage.s <- as.vector(switchedPredictions[, "Actual"])
sn <- length(actual.stage.s)

for(i in 4:length(switchedPredictions)){
  sv <- as.vector(switchedPredictions[,i])
  switched[i-3] <- round(100*length(which(actual.stage.s == sv))/sn, 3)
}

allPrecisions <- data.frame(classifier.name, unswitched, switched)
allPrecisions<- melt(allPrecisions, id.vars = "classifier.name")
precision.plot <- ggplot(allPrecisions, aes(y = value, x = classifier.name, fill = variable))
precision.plot <- precision.plot + geom_bar(stat = "identity", position = "dodge")
precision.plot <- precision.plot + xlab("Classifier Name") + ylab("Percent Precision")
precision.plot <- precision.plot + guides(fill=guide_legend(title = "Training Data"))
precision.plot <- precision.plot + scale_fill_manual(values = c("#0033FF", "#6699FF"))
precision.plot <- precision.plot + geom_text(aes(label = round(value, 1)), position = position_dodge(0.75), vjust = -1)
precision.plot <- precision.plot + theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 24))
precision.plot

### Finding out which algorithms performed better between switched data
###BETTER UNSWITCHED
classifier.name[which(unswitched > switched)]

###BETTER SWITCHED
classifier.name[which(unswitched < switched)]

### RESULTS WHEN ATTRIBUTES ARE RANKED FIRST
rankedPredictions <- read.csv("RankedPredictions.csv", header = TRUE)
ranked <- rep(0, length(rankedPredictions) - 3)
classifier.name.r <- colnames(rankedPredictions[4:length(rankedPredictions)])
actual.stage.r <- as.vector(rankedPredictions[, "Actual"])
rn <- length(actual.stage.r)

for(i in 4:length(rankedPredictions)){
  rv <- as.vector(rankedPredictions[,i])
  ranked[i-3] <- round(100*length(which(actual.stage.r == rv))/rn, 3)
}

allPrecisions <- data.frame(classifier.name.r, switched, ranked)
allPrecisions<- melt(allPrecisions, id.vars = "classifier.name.r")
precision.plot <- ggplot(allPrecisions, aes(y = value, x = classifier.name.r, fill = variable))
precision.plot <- precision.plot + geom_bar(stat = "identity", position = "dodge")
precision.plot <- precision.plot + xlab("Classifier Name") + ylab("Percent Precision")
precision.plot <- precision.plot + guides(fill=guide_legend(title = "Training Data"))
precision.plot <- precision.plot + scale_fill_manual(values = c("#0033FF", "#6699FF"))
precision.plot <- precision.plot + geom_text(aes(label = round(value, 1)), position = position_dodge(0.75), vjust = -1)
precision.plot <- precision.plot + theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 24))
precision.plot

###BETTER RANKED
classifier.name.r[which(ranked > switched)]

###NO DIFFERENCE
classifier.name.r[which(ranked == switched)]

###BETTER UNRANKED
classifier.name.r[which(ranked < switched)]

###LOOKING AT THE STAGE PREDICTION BREAKDOWN WHEN THE ATTRIBUTES ARE RANKED
predictions <- read.csv("RankedPredictions.csv", header = TRUE)
classifier.name <- colnames(predictions[5:length(predictions)])
actual.stage <- as.vector(predictions[, "Actual"])
n <- length(actual.stage)

stages <- c("Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5")
numStages <- rep(0, length(stages))
for (i in 1:length(stages)){
  numStages[i] <- length(which(predictions[,"Actual"] == stages[i]))
}

percent.correct <- rep(0, length(stages))

for(i in 1:length(stages)){
  indices <- which(predictions[,"Actual"] == stages[i])
  for(j in 5:length(predictions)){
    percent.correct[i] <- percent.correct[i] + length(which(predictions[indices, j] == stages[i]))
  }
}

### percent.correct is an AVERAGE of the percent time a stage is predicted correctly
percent.correct <- 100*(round(percent.correct/(numStages*(length(classifier.name)-1)),3))
percent.correct <- data.frame(stages, percent.correct)

stage.plot <- ggplot(percent.correct, aes(y = percent.correct, x = stages))
stage.plot <- stage.plot + geom_bar(stat = "identity", fill = "#0033FF")
stage.plot <- stage.plot + xlab("Stage") + ylab("Percent Correctly Predicted") + ggtitle("Average Precision per Stage")
stage.plot <- stage.plot + geom_text(aes(label = round(percent.correct, 1)), position = position_dodge(0.75), vjust = -1)
stage.plot

###What percentage of the time was stage 1 predicted?
times.predicted <- rep(0, length(stages))

for(i in 1:length(stages)){
  for(j in 5:length(predictions)){
    times.predicted[i] <- times.predicted[i] + length(which(predictions[,j] == stages[i]))
  }
}

times.predicted <- 100*(round(times.predicted/(sum(times.predicted)),5))
times.predicted <- data.frame(stages, times.predicted)
times.predicted.plot <- ggplot(times.predicted, aes(y = times.predicted, x = stages))
times.predicted.plot <- times.predicted.plot + geom_bar(stat = "identity", fill = "#0033FF")
times.predicted.plot <- times.predicted.plot + xlab("Stage") + ylab("Percent of Time Predicted") + ggtitle("Percentage Predicted per Stage")
times.predicted.plot <- times.predicted.plot + geom_text(aes(label = round(times.predicted, 1)), position = position_dodge(0.75), vjust = -1)
times.predicted.plot

### WORKING WITH JUST THE SWITCHED DATA SINCE THIS LED TO OVERALL BETTER RESULTS
predictions <- read.csv("switchedPredictions.csv", header = TRUE)
classifier.name <- colnames(predictions[5:length(predictions)])
actual.stage <- as.vector(predictions[, "Actual"])
n <- length(actual.stage)

stages <- c("Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5")
numStages <- rep(0, length(stages))
for (i in 1:length(stages)){
  numStages[i] <- length(which(predictions[,"Actual"] == stages[i]))
}

percent.correct <- rep(0, length(stages))

for(i in 1:length(stages)){
  indices <- which(predictions[,"Actual"] == stages[i])
  for(j in 5:length(predictions)){
   percent.correct[i] <- percent.correct[i] + length(which(predictions[indices, j] == stages[i]))
  }
}

### percent.correct is an AVERAGE of the percent time a stage is predicted correctly
percent.correct <- 100*(round(percent.correct/(numStages*(length(classifier.name)-1)),3))
percent.correct <- data.frame(stages, percent.correct)

stage.plot <- ggplot(percent.correct, aes(y = percent.correct, x = stages))
stage.plot <- stage.plot + geom_bar(stat = "identity", fill = "#0033FF")
stage.plot <- stage.plot + xlab("Stage") + ylab("Percent Correctly Predicted")
stage.plot <- stage.plot + geom_text(aes(label = round(percent.correct, 1)), position = position_dodge(0.75), vjust = -1)
stage.plot <- stage.plot + theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 18))
stage.plot

###What percentage of the time was stage 1 predicted?
predictions <- read.csv("switchedPredictions.csv", header = TRUE)
times.predicted <- rep(0, length(stages))

for(i in 1:length(stages)){
  for(j in 5:length(predictions)){
    times.predicted[i] <- times.predicted[i] + length(which(predictions[,j] == stages[i]))
  }
}

times.predicted <- 100*(round(times.predicted/(sum(times.predicted)),5))
times.predicted <- data.frame(stages, times.predicted)
times.predicted.plot <- ggplot(times.predicted, aes(y = times.predicted, x = stages))
times.predicted.plot <- times.predicted.plot + geom_bar(stat = "identity", fill = "#0033FF")
times.predicted.plot <- times.predicted.plot + xlab("Stage") + ylab("Percent of Time Predicted")
times.predicted.plot <- times.predicted.plot + geom_text(aes(label = round(times.predicted, 1)), position = position_dodge(0.75), vjust = -1)
times.predicted.plot <- times.predicted.plot + theme(axis.text.x = element_text(angle = 45, hjust = 1), text = element_text(size = 18))
times.predicted.plot

###How many of each stage was there in the testing set?
trainingSet <- read.csv("DatabaseTesting.csv", header = TRUE)
stages.training <- rep(0, length(stages))
for (i in 1:length(stages)){
  stages.training[i] <- length(which(trainingSet[,"Stage"] == stages[i]))
}


####WORKING WITH THE ORIGINAL DATA, COPIED CODE BASICALLY FROM ABOVE
predictions <- read.csv("originalPredictions.csv", header = TRUE)
classifier.name <- colnames(predictions[5:length(predictions)])
actual.stage <- as.vector(predictions[, "Actual"])
n <- length(actual.stage)

stages <- c("Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5")
numStages <- rep(0, length(stages))
for (i in 1:length(stages)){
  numStages[i] <- length(which(predictions[,"Actual"] == stages[i]))
}

percent.correct <- rep(0, length(stages))

for(i in 1:length(stages)){
  indices <- which(predictions[,"Actual"] == stages[i])
  for(j in 5:length(predictions)){
    percent.correct[i] <- percent.correct[i] + length(which(predictions[indices, j] == stages[i]))
  }
}

### percent.correct is an AVERAGE of the percent time a stage is predicted correctly
percent.correct <- 100*(round(percent.correct/(numStages*(length(classifier.name)-1)),3))
percent.correct <- data.frame(stages, percent.correct)

stage.plot <- ggplot(percent.correct, aes(y = percent.correct, x = stages))
stage.plot <- stage.plot + geom_bar(stat = "identity", fill = "#0033FF")
stage.plot <- stage.plot + xlab("Stage") + ylab("Percent Correctly Predicted") + ggtitle("Average Precision per Stage")
stage.plot <- stage.plot + geom_text(aes(label = round(percent.correct, 1)), position = position_dodge(0.75), vjust = -1)
stage.plot

###What percentage of the time was stage 1 predicted?
times.predicted <- rep(0, length(stages))

for(i in 1:length(stages)){
  for(j in 5:length(predictions)){
    times.predicted[i] <- times.predicted[i] + length(which(predictions[,j] == stages[i]))
  }
}

times.predicted <- 100*(round(times.predicted/(sum(times.predicted)),5))
times.predicted <- data.frame(stages, times.predicted)
times.predicted.plot <- ggplot(times.predicted, aes(y = times.predicted, x = stages))
times.predicted.plot <- times.predicted.plot + geom_bar(stat = "identity", fill = "#0033FF")
times.predicted.plot <- times.predicted.plot + xlab("Stage") + ylab("Percent of Time Predicted") + ggtitle("Percentage Predicted per Stage")
times.predicted.plot <- times.predicted.plot + geom_text(aes(label = round(times.predicted, 1)), position = position_dodge(0.75), vjust = -1)
times.predicted.plot