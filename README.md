# BioInformatics

Weka Machine Learning Classifier Application
-Program automatically queries the McGill Phenomics database and creates .arff files
-These .arff files are then read by the program and used to train and test WEKA classification algorithms based on which algorithms are inputed by the user
-Output produced are two CSV's
	- The first CSV shows the barcode of the plant, its actual developmental stage, and the developmental stage predicted by the classification algorithm
	- The second CSV shows the accuracy of each classification algorithm, produced to then be able to graph outputs in R
