
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define TPB 32

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num_elements) {
        return;
    }
    unsigned val = input[i];
    atomicAdd(bins + val, 1);
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num_bins) {
        return;
    }

    if(bins[i] > 127) {
        bins[i] = 127;
    }
}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned *) malloc(inputLength * sizeof(unsigned));
  hostBins = (unsigned *) malloc(NUM_BINS * sizeof(unsigned));
  resultRef = (unsigned *) malloc(NUM_BINS * sizeof(unsigned));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::random_device rd;
  std::mt19937 gen(rd());
  // Still the same with floats. Rounding down to ints still gives a uniform distribution as NUM_BINS is exclusive
  auto dis = std::uniform_real_distribution<float>(0, NUM_BINS);
  for(int i = 0; i < inputLength; i++) {
      hostInput[i] = (int) dis(gen);
  }

  //@@ Insert code below to create reference result in CPU
  for(int i = 0; i < inputLength; i++) {
      resultRef[hostInput[i]]++;
  }
  for(int i = 0; i < NUM_BINS; i++) {
      if(resultRef[i] > 127) {
          resultRef[i] = 127;
      }
  }
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  //TODO ?

  //@@ Initialize the grid and block dimensions here
  int Db1 = TPB;
  int Dg1 = (inputLength + Db1 - 1) / Db1;


  //@@ Launch the GPU Kernel here
  histogram_kernel<<<Dg1, Db1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);


  //@@ Initialize the second grid and block dimensions here
  int Db2 = TPB;
  int Dg2 = (NUM_BINS + Db2 - 1) / Db2;


  //@@ Launch the second GPU Kernel here
  convert_kernel<<<Dg2, Db1>>>(deviceBins, NUM_BINS);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  for(int i = 0; i < NUM_BINS; i++) {
      unsigned res_cpu = resultRef[i];
      unsigned res_gpu = hostBins[i];
      if(res_gpu != res_cpu) {
          printf("%i wrong: %u - %u\n", i, res_gpu, res_cpu);
      } else {
          // printf("Accurate result\n");
          fprintf(stderr, "%i;%i\n", i, res_gpu);
      }
  }


  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);


  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);


  return 0;
}

