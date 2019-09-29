#ifndef HELPER_H
#define HELPER_H

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
	mat = (float*)calloc(m*n, sizeof(float));

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

#endif