#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "helper.h"
#include "mergeSort.h"

#define NTRAIN 135
#define NTEST 15
#define NFEATURES 4
#define NCLASSES 3

char class[NCLASSES][25] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

void mpiInitialise(int *size, int *rank)
{
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, size);
}

float *initFeatures(char path[])
{
	int index = 0;
	FILE *f  = NULL;
	float *mat = NULL;

	mat = getFloatMat(NTRAIN, NFEATURES);

	f = fopen(path, "r");
	checkFile(f);

	while (fscanf(f, "%f%*c", &mat[index]) == 1) //%*c ignores the comma while reading the CSV
		index++;

	fclose(f);
	return mat;
}

float *initLabels(char path[])
{
	int index = 0;
	FILE *f  = NULL;
	float *mat = NULL;

	mat = getFloatMat(NTRAIN, 1);

	f = fopen(path, "r");
	checkFile(f);

	while (fscanf(f, "%f%*c", &mat[index]) == 1)
		index++;

	fclose(f);
	return mat;
}

int predict(float *distance, float *labels, int k, int topn) //topn < NCLASSES
{
	float* neighborCount = getFloatMat(NCLASSES, 1);
	float* probability = getFloatMat(NCLASSES, 1);

	int i;

	for(i=0; i<k; i++)
		neighborCount[(int)labels[i]]++;

	for(i=0; i<NCLASSES; i++)
		probability[i] = neighborCount[i]*1.0/(float)k*1.0;
	
	int predicted_class = (int)getMax(neighborCount, NCLASSES);

	printf("Probability:\n");
	for(i=0; i<topn; i++)
		printf("%s\t%f\n", class[i], probability[i]);

	free(neighborCount);
	free(probability);

	return predicted_class;
}

void calcDistance(int ndata_per_process, float *pdistance, float *pdata, float *x)
{
	int index = 0, i, j;
	for(i=0; i<ndata_per_process; i=i+NFEATURES)
	{
		pdistance[index] = 0.0;

		for(j=0; j<NFEATURES; j++)
			pdistance[index] = pdistance[index] + (pdata[i+j]-x[j])*(pdata[i+j]-x[j]);

		index++;
	}
}

void fit(float *X_train, float *y_train, float *X_test, float *y_test, int rank, int size)
{
	int i, j;
	int ndata_per_process, nrows_per_process;
	float *pdata, *distance, *pdistance;
	float *plabels;
	float *labels;

	if (NTRAIN % size != 0)
	{
		if (rank == 0)
			printf("Not divisible\n");
		
		MPI_Finalize();
		exit(0);
	}

	// initialise arrays
	nrows_per_process = NTRAIN/size;
	ndata_per_process = nrows_per_process*NFEATURES;

	pdata = getFloatMat(ndata_per_process, 1);
	pdistance = getFloatMat(nrows_per_process, 1);
	distance = getFloatMat(NTRAIN, 1);

	plabels = getFloatMat(nrows_per_process, 1);
	labels = getFloatMat(NTRAIN, 1);

	MPI_Scatter(X_train, ndata_per_process, MPI_FLOAT, pdata, ndata_per_process, MPI_FLOAT, 0,  MPI_COMM_WORLD);

	float *x = getFloatMat(NFEATURES, 1);

	for (i=0; i<NTEST; i=i+1)
	{	
		// very imp to scatter everytime in the loop here since plabels keep getting sorted and associativity is changed. 
		MPI_Scatter(y_train, nrows_per_process, MPI_FLOAT, plabels, nrows_per_process, MPI_FLOAT, 0,  MPI_COMM_WORLD);

		for(j=0; j<NFEATURES; j++)
			x[j] = X_test[i*NFEATURES+j];


		// fit
		calcDistance(ndata_per_process, pdistance, pdata, x);

		//sort the distance array 
		mergeSort(pdistance, 0, nrows_per_process - 1, plabels);

		MPI_Gather(pdistance, nrows_per_process, MPI_FLOAT, distance, nrows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Gather(plabels, nrows_per_process, MPI_FLOAT, labels, nrows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);


		if (rank == 0)
		{
			mergeSort(distance, 0, NTRAIN - 1, labels);
			int predicted_class = predict(distance, labels, 5, 3);
			printf("%d) Predicted label: %d   True label: %d\n\n", i, predicted_class, (int)y_test[i]);
		}
	}

	free(x);
	free(distance);
	free(pdistance);
}

void knn(char *X_train_path, char *y_train_path, char *X_test_path, char *y_test_path)
{
	float *X_train;
	float *y_train;
	float *X_test;
	float *y_test;
	double t1, t2;
	int size, rank;

	mpiInitialise(&size, &rank);

	if (rank == 0)
	{
		X_train = initFeatures(X_train_path);
		y_train = initLabels(y_train_path);
	}

	X_test = initFeatures(X_test_path);
	y_test = initLabels(y_test_path);

	if (rank == 0)
		t1 = MPI_Wtime();

	fit(X_train, y_train, X_test, y_test, rank, size);

	if (rank == 0)
		t2 = MPI_Wtime();

	if (rank == 0)
	{
		printf("Time for execution (%d Processors): %f\n", size, t2 - t1);
		free(X_train);
		free(y_train);
	}

	free(X_test);
	free(y_test);
	MPI_Finalize();
}

int main()
{
	knn("X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv");
	return 0;
}