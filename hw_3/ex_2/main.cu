
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define DataType double
#define TPB 32

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= numARows * numBColumns) {
        return;
    }
  int numCRows = numARows;
  int numCColumns = numBColumns;

    int c = i % numCColumns;
    int r = i / numCColumns;
    DataType res = 0.0;

    for(int i = 0; i < numAColumns; i++) {
        res += A[r * numAColumns + i] * B[c + i * numBColumns];
    }
    C[r * numCColumns + c] = res;

}

//@@ Insert code to implement timer start
long startTimer() {
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    return timecheck.tv_sec * 1000000 + timecheck.tv_usec;
}


//@@ Insert code to implement timer stop
long endTimer(long start) {
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    long end = (long) timecheck.tv_sec * 1000000 + (long) timecheck.tv_usec;
    return end - start;
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = numAColumns;
  numBColumns = atoi(argv[3]);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *) malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *) malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *) malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *) malloc(numCRows * numCColumns * sizeof(DataType));

  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dis = std::uniform_real_distribution<DataType>(0.0, 1.0);
  for(int r = 0; r < numARows; r++) {
      for(int c = 0; c < numAColumns; c++) {
          // Row-major
          DataType val = dis(gen);
          hostA[r * numAColumns + c] = val;
          printf("%.1f ", val);
      }
      printf("\n");
  }
  printf("\n");
  for(int r = 0; r < numBRows; r++) {
      for(int c = 0; c < numBColumns; c++) {
          // Row-major
          DataType val = dis(gen);
          hostB[r * numBColumns + c] = val;
          printf("%.1f ", val);
      }
      printf("\n");
  }
  for(int r = 0; r < numCRows; r++) {
      for(int c = 0; c < numCColumns; c++) {
          // Row-major
          DataType res = 0.0;

          for(int i = 0; i < numAColumns; i++) {
              res += hostA[r * numAColumns + i] * hostB[c + i * numBColumns];
          }
          resultRef[r * numCColumns + c] = res;
      }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  long start = startTimer();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  printf("Copying to device in %li mus\n", endTimer(start));


  //@@ Initialize the grid and block dimensions here
  int Db = TPB;
  int Dg = (numCRows * numCColumns + Db - 1) / Db;


  //@@ Launch the GPU Kernel here
  start = startTimer();
  gemm<<<Dg, Db>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  printf("Multiplying in %li mus\n", endTimer(start));


  //@@ Copy the GPU memory back to the CPU here
  start = startTimer();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("Copying from device in %li mus\n", endTimer(start));

  //@@ Insert code below to compare the output with the reference
  for(int r = 0; r < numCRows; r++) {
      for(int c = 0; c < numCColumns; c++) {
          DataType res_cpu = resultRef[r * numCColumns + c];
          DataType res_gpu = hostC[r * numCColumns + c];
          if(std::abs(res_gpu - res_cpu) > 0.001) {
              printf("Deviation too large\n%f - %f\n", res_gpu, res_cpu);
          } else {
              // printf("Accurate result\n");
              printf("%.1f ", res_gpu);
          }
      }
      printf("\n");
  }


  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);


  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);


  return 0;
}
