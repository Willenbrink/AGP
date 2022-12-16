#include <stdio.h>
#include <sys/time.h>
#include <random>

#define DataType double

#define TPB 32

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= len) {
            return;
    }
    out[i] = in1[i] + in2[i];
    // printf("1: %f 2: %f O: %f\n", in1[i], in2[i], out[i]);
}

long start, end;
//@@ Insert code to implement timer start
void startTimer() {
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    start = (long) timecheck.tv_sec * 1000000 + (long) timecheck.tv_usec;
}


//@@ Insert code to implement timer stop
long endTimer() {
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    end = (long) timecheck.tv_sec * 1000000 + (long) timecheck.tv_usec;
    return end - start;
}


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  

  printf("The input length is %i\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *) malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *) malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *) malloc(inputLength * sizeof(DataType));
  resultRef = (DataType *) malloc(inputLength * sizeof(DataType));
  
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dis = std::uniform_real_distribution<DataType>(0.0, 1.0);
  for(int i = 0; i < inputLength; i++) {
      hostInput1[i] = dis(gen);
      hostInput2[i] = dis(gen);
      resultRef[i] = hostInput1[i] + hostInput2[i];
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  startTimer();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  printf("Copying to device in %li mus\n", endTimer());

  //@@ Initialize the 1D grid and block dimensions here
  int Db = TPB;
  int Dg = (inputLength + Db - 1) / Db;

  //@@ Launch the GPU Kernel here
  startTimer();
  vecAdd<<<Dg, Db>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  printf("Adding in %li mus\n", endTimer());

  //@@ Copy the GPU memory back to the CPU here
  startTimer();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("Copying from device in %li mus\n", endTimer());

  //@@ Insert code below to compare the output with the reference
  for(int i = 0; i < inputLength; i++) {
      DataType res_cpu = resultRef[i];
      DataType res_gpu = hostOutput[i];
      if(std::abs(res_gpu - res_cpu) > 0.001) {
          printf("Deviation too large\n%f - %f\n", res_gpu, res_cpu);
      } else {
          // printf("Accurate result\n");
      }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
