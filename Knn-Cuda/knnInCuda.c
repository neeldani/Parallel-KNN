#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// cofig file, make changes here
#include "config.h"

void checkFile(FILE *f)
{
	if (f == NULL)
	{
		printf("Error while reading file\n");
		exit(1);
	}
}


float *getFloatMat(int m, int n)
{
	float *mat = NULL;
	mat = (float*)malloc(m*n*sizeof(float));

	return mat;
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

float getMax(float *x, int n)
{
	int i;
	float max = x[0];
	int maxIndex = 0;

	for(i=0; i<n; i++)
	{
		if (x[i] >= max)
		{
			max = x[i];
			maxIndex = i;
		}
	}

	return (float)maxIndex;
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


__global__ void calcDistance (float *X_train, float *X_test, float *distance)
{
    int blockid = blockIdx.x;
    int threadid = blockDim.x*blockid + threadIdx.x;
    
    if (threadid < NTRAIN)
    {
        int i;
        float dist = 0.0;
    
        for(i=0; i<NFEATURES; i++)
        {
            dist += (X_train[threadid*NFEATURES + i] - X_test[i])*(X_train[threadid*NFEATURES + i] - X_test[i]);
        }
    
        distance[threadid] = dist;
    }
}

__global__ void sortArray (float *distance, float *ytrain, float *sortedDistance, float *sortedYtrain)
{
    int blockid = blockIdx.x;
    int threadid = blockDim.x*blockid + threadIdx.x;
    
    if (threadid < NTRAIN)
    {
        int i, position = 0;
        float element = distance[threadid];
        float label = ytrain[threadid];
    
        for(i=0; i<NTRAIN; i++)
        {
            if (distance[i] < element || (distance[i] == element && threadid < i) )
                position++;
        }
    
        sortedDistance[position] = element;
        sortedYtrain[position] = label;
    }
}

int predict(float *labels)
{
	float* neighborCount = getFloatMat(NCLASSES, 1);
    
	float* probability = getFloatMat(NCLASSES, 1);

	int i;
    for(i=0; i<NCLASSES; i++)
        neighborCount[i] = 0;

	for(i=0; i<K; i++)
		neighborCount[(int)labels[i]]++;

	for(i=0; i<NCLASSES; i++)
		probability[i] = neighborCount[i]*1.0/(float)K*1.0;
	
	int predicted_class = (int)getMax(neighborCount, NCLASSES);

	for(i=0; i<TOPN; i++)
		printf(" %s: %f ", classes[i], probability[i]);

	free(neighborCount);
	free(probability);

	return predicted_class;
}

float *fit(float *X_train, float *y_train, float *X_test)
{
    float *X_traind, *y_traind, *X_testd, *distanced, *distance;
    
    distance = getFloatMat(NTRAIN, 1);
    
    int X_train_size = sizeof(float)*NFEATURES*NTRAIN;
    int y_train_size = sizeof(float)*NTRAIN;
    int X_test_size = sizeof(float)*NFEATURES;
    int distance_size = sizeof(float)*NTRAIN;
    
    
    cudaMalloc((void**)&X_traind, X_train_size);
    cudaMalloc((void**)&y_traind, y_train_size);
    cudaMalloc((void**)&X_testd, X_test_size);
    cudaMalloc((void**)&distanced, distance_size);
    
    cudaMemcpy(X_traind, X_train, X_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_traind, y_train, y_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(X_testd, X_test, X_test_size, cudaMemcpyHostToDevice);
   
    //launch distance kernel 
    calcDistance <<< NTRAIN/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (X_traind, X_testd, distanced); 
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(distance, distanced, distance_size, cudaMemcpyDeviceToHost);
    
    cudaFree(X_traind);
    cudaFree(X_testd);
    
    /* Launching a kernel to sort the distances and make corresponing swaps in the y_train */ 
    float *sortedDistance, *sortedDistanced, *sortedytrain, *sortedytraind;
       
    sortedDistance = getFloatMat(NTRAIN, 1);
    sortedytrain = getFloatMat(NTRAIN, 1);
    
    cudaMalloc((void**)&sortedDistanced, distance_size);
    cudaMalloc((void**)&sortedytraind, y_train_size);
    
    cudaMemcpy(distanced, distance, distance_size, cudaMemcpyHostToDevice);
    
    //call sorting kernel
    sortArray <<< NTRAIN/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (distanced, y_traind, sortedDistanced, sortedytraind);
    
    cudaMemcpy(sortedDistance, sortedDistanced, distance_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(sortedytrain, sortedytraind, y_train_size, cudaMemcpyDeviceToHost);
    
    cudaFree(y_traind);
    cudaFree(distanced);
    cudaFree(sortedDistanced);
    cudaFree(sortedytraind);
    
    free(distance);
    free(sortedDistance);
    
    return sortedytrain;
}

float *getRandomTestData(float *X_test, int *randId)
{
    srand ( time(NULL) );
    *randId = rand()%NTEST;
    float *data = getFloatMat(NFEATURES, 1);
    
    int i;
    for(i=0; i<NFEATURES; i++)
        data[i] = X_test[(*randId)*NFEATURES + i];
    
    return data;
}

void readData(float **X_train, float **y_train, float **X_test, float **y_test)
{
    *X_train = initFeatures(X_TRAIN_PATH);
	*y_train = initLabels(Y_TRAIN_PATH);

	*X_test = initFeatures(X_TEST_PATH);
	*y_test = initLabels(Y_TEST_PATH);
}

int knn(float *X_train, float *y_train, float *X_test)
{
    
    printf(" Fitting model ");
    
    float et;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
	float *labels = fit(X_train, y_train, X_test);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
   
    printf("Time taken: %fms", et);
    
    int predicted_class = predict(labels);
    return predicted_class;
}

int main()
{
    float *X_train;
	float *y_train;
	float *X_test;
	float *y_test;
 
    
    //read data
    readData(&X_train, &y_train, &X_test, &y_test);
    
    int randId;
    float *X_random_test = getRandomTestData(X_test, &randId);
    
    //call knn
    int predicted_class = knn(X_train, y_train, X_random_test);
    
    
    printf("Predicted label: %d True label: %d", predicted_class, (int)y_test[randId]);
    
     
	free(X_train);
	free(y_train);

	free(X_test);
	free(y_test);
    
    return 0;
}
