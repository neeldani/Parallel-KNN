#define NTRAIN 135						// Number of training examples
#define NTEST 15						// Number of testing examples
#define NFEATURES 4						// Number of features (columns) in th each training example
#define NCLASSES 3						// Number of labels/ classes
#define K 11							// Hyperparameter K in KNN
#define TOPN 3							// Get the top N predictions
#define THREADS_PER_BLOCK 5				// Number of threads used in CUDA for each block 
#define X_TRAIN_PATH "X_train.csv"		// Path to X train .csv file which contains train data
#define Y_TRAIN_PATH "y_train.csv"		// Path to y train .csv which contains labels for X train
#define X_TEST_PATH "X_test.csv"		// Path to X test .csv file which contains test data
#define Y_TEST_PATH "y_test.csv"		// Path to y test .csv file which conatins labels for y train


// Array containing list of labels. Make changes 
char classes[NCLASSES][25] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};