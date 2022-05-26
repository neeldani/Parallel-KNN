# Parallel-KNN
A Parallel implementation of the famous K-Nearest-Neighbor (KNN) algorithm in CUDA as well as MPI.

<h2> Dependencies </h2>

For KNN in MPI:
<ul>
  <li>gcc-5.4.0</li>
  <li>openmpi-1.10.2</li>
</ul>

For KNN in CUDA:
<ul>
  <li>nvcc release 9.2 </li>
</ul>

<h2> Usage </h2>

Clone the repository using:
```console
foo@bar:~$ git clone https://github.com/neeldani/Parallel-KNN.git
```

Navigate to the project directory:

**For MPI**:
```console
foo@bar:~$ cd Knn-CUDA 
```
Compile and execute the MPI code:
```console
foo@bar:~$ mpicc knnInMPI.c -o knnInMpi.out -Wall
foo@bar:~$ mpirun -n 3 knnInMpi.out
```

**For CUDA**:
```console
foo@bar:~$ cd Knn-MPI 
```

Compile and execute the CUDA code:
```console
foo@bar:~$ nvcc -o Knn-Cuda.out Knn-Cuda.cu 
```

<h2> Configuration </h2>

The config.h file in MPI and CUDA can be used to set hyperparameters and the path to the dataset. 
The dataset should be split in a .csv file having individual files for training examples (X_train), training labels (y_train), testing examples (X_test) and testing labels (y_test). An example of the sample data (Iris Dataset) is present in the cloned repoisitory.

<h2> Future Works </h2>

Currently, the code is functional only for matrices having number of rows divisible by number of processes (in MPI) or number of threads (in CUDA) (both can be configured in the config.h file). Future works include generalization of the algorithm by allowing any dimension matrices and designing of thee algorithm to minimize the overhead due to communication.
