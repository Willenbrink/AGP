#include <stdio.h>
#include <sys/time.h>
#include <random>

#define DataType float

#define TPB 32

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= len) {
            return;
    }
    // for(int j = 0; j < 100; j++)
    {
        out[i] = in1[i] + in2[i];
    }
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
  int n_seg = atoi(argv[2]);
  

  // int n_seg = (inputLength + s_seg - 1) / s_seg;
  // int s_seg = 1<<15;
  int s_seg = (inputLength + n_seg - 1) / n_seg;
  int total = s_seg * n_seg;

  printf("The input length is %i\n\tn_seg: %i\n\ts_seg: %i\n", inputLength, n_seg, s_seg);

  //@@ Insert code below to allocate Host memory for input and output
  // cudaHostAlloc(&hostInput1, inputLength * sizeof(DataType), cudaHostAllocDefault);
  // cudaHostAlloc(&hostInput2, inputLength * sizeof(DataType), cudaHostAllocDefault);
  // cudaHostAlloc(&hostOutput, inputLength * sizeof(DataType), cudaHostAllocDefault);
  hostInput1 = (DataType *) malloc(total * sizeof(DataType));
  hostInput2 = (DataType *) malloc(total * sizeof(DataType));
  hostOutput = (DataType *) malloc(total * sizeof(DataType));

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

  cudaMalloc(&deviceInput1, total * sizeof(DataType));
  cudaMalloc(&deviceInput2, total * sizeof(DataType));
  cudaMalloc(&deviceOutput, total * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  cudaStream_t streams[4];
  for(int i = 0; i < 4; i++)
      cudaStreamCreate(streams + i);

  startTimer();
  printf("Running %i times\n", n_seg);
  for(int i = 0; i < n_seg; i++) {
      int off = i * s_seg;
      int stream;
      stream = i % 4;
      cudaMemcpyAsync(deviceInput1 + off, hostInput1 + off, s_seg * sizeof(DataType), cudaMemcpyHostToDevice, streams[stream]);
      cudaMemcpyAsync(deviceInput2 + off, hostInput2 + off, s_seg * sizeof(DataType), cudaMemcpyHostToDevice, streams[stream]);
      // cudaMemcpy(deviceInput1 + off, hostInput1 + off, S_seg * sizeof(DataType), cudaMemcpyHostToDevice);
      // cudaMemcpy(deviceInput2 + off, hostInput2 + off, S_seg * sizeof(DataType), cudaMemcpyHostToDevice);
  // }

  // for(int i = 0; i < n_seg; i++) {
  //     int off = i * s_seg;
  //     int stream;
  //     stream = i % 4;
      int Db = TPB;
      int Dg = (s_seg + Db - 1) / Db;
      vecAdd<<<Dg, Db, 0, streams[stream]>>>(deviceInput1 + off, deviceInput2 + off, deviceOutput + off, inputLength - off);
  }

  for(int i = 0; i < n_seg; i++) {
      int off = i * s_seg;
      int stream;
      stream = i % 4;
      cudaMemcpyAsync(hostOutput + off, deviceOutput + off, s_seg * sizeof(DataType), cudaMemcpyDeviceToHost, streams[stream]);
  }
  printf("Done in %li mus\n", endTimer());

  //@@ Insert code below to compare the output with the reference
  // for(int i = 0; i < inputLength; i++) {
  //     DataType res_cpu = resultRef[i];
  //     DataType res_gpu = hostOutput[i];
  //     if(std::abs(res_gpu - res_cpu) > 0.001) {
  //         printf("Deviation too large\n%f - %f\n", res_gpu, res_cpu);
  //     } else {
  //         // printf("Accurate result\n");
  //     }
  // }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  free(resultRef);

  for(int i = 0; i < 4; i++)
      cudaStreamDestroy(streams[i]);

  return 0;
}
