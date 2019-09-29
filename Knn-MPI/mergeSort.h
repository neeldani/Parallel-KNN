#ifndef M_SORT
#define M_SORT

void merge(float arr[], int l, int m, int r, float *y) 
{ 
    int i, j, k; 
    int n1 = m - l + 1; 
    int n2 =  r - m; 
  
    /* create temp arrays */
    float *L = getFloatMat(n1, 1);
    float *R = getFloatMat(n2, 1); 

    float *Ly = getFloatMat(n1, 1);
    float *Ry = getFloatMat(n2, 1);
  
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
    {
        L[i] = arr[l + i];
        Ly[i] = y[l + i];
    }

    for (j = 0; j < n2; j++)
    {
        R[j] = arr[m + 1+ j];
        Ry[j] = y[m + 1 + j];
    } 
  
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray 
    j = 0; // Initial index of second subarray 
    k = l; // Initial index of merged subarray 
    while (i < n1 && j < n2) 
    { 
        if (L[i] <= R[j]) 
        { 
            arr[k] = L[i];
            y[k] = Ly[i];
            i++; 
        } 

        else
        { 
            arr[k] = R[j]; 
            y[k] = Ry[j];

            j++; 
        }

        k++; 
    } 
  
    /* Copy the remaining elements of L[], if there 
       are any */
    while (i < n1) 
    { 
        arr[k] = L[i];
        y[k] = Ly[i];

        i++; 
        k++; 
    } 
  
    /* Copy the remaining elements of R[], if there 
       are any */
    while (j < n2) 
    { 
        arr[k] = R[j]; 
        y[k] = Ry[j];

        j++; 
        k++; 
    } 

    free (L);
    free (R);

    free (Ly);
    free (Ry);
} 
  
/* l is for left index and r is right index of the 
   sub-array of arr to be sorted */

void mergeSort(float arr[], int l, int r, float *y) 
{ 
    if (l < r) 
    { 
        // Same as (l+r)/2, but avoids overflow for 
        // large l and h 
        int m = l+(r-l)/2; 
  
        // Sort first and second halves 
        mergeSort(arr, l, m, y); 
        mergeSort(arr, m+1, r, y); 
  
        merge(arr, l, m, r, y); 
    } 
} 

void printArray(float A[], int size) 
{ 
    int i; 
    for (i=0; i < size; i++) 
        printf("%f ", A[i]); 
    printf("\n"); 
} 

#endif