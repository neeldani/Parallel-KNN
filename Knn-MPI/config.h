#ifndef CONFIG_H
#define CONFIG_H

#define NTRAIN 135 						// number of training examples
#define NTEST 15 						// number of testing examples 
#define NFEATURES 4 					// number of features (columns) in te training examples
#define NCLASSES 3 						// number of classes/ labels 

#define X_TRAIN_PATH "../X_train.csv" 	// path to X train .csv file
#define Y_TRAIN_PATH "../y_train.csv" 	// path to y train .csv file
#define X_TEST_PATH "../X_test.csv"	  	// path to X test .csv file
#define Y_TEST_PATH "../y_test.csv"   	// path to y test .csv file

#define K 3 						  	// the hyperparameter K in KNN algorithm
#define TOPN 2 					      	// Print the closest top N classes


// Array of all classes/ label names 
char class[NCLASSES][25] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};


#endif