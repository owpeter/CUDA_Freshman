#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"


void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}
__global__ void sumArraysGPU(float*a,float*b,float*res)
{
  //int i=threadIdx.x;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  // blockIdx.x: 当前线程所在的block在grid中的索引（从0开始）
  // blockDim.x: 每个block中的线程数量（在你的代码中是1024）。
  // threadIdx.x: 当前线程在block内的局部索引（0到1023）。
  // 得到的i是当前线程的全局唯一索引
  res[i]=a[i]+b[i];
  printf("[%d]\t\tres: %f\n", i, res[i]);
}
// Main function with command line arguments
int main(int argc, char **argv) 
{
    // Set up CUDA device
    int dev = 0;  // Use device 0 (the first GPU)
    cudaSetDevice(dev);  // Set the current CUDA device

    // Define vector size and calculate memory requirements
    int nElem = 1 << 14;  // Vector size = 2^14 = 16384 elements
    printf("Vector size: %d\n", nElem);
    int nByte = sizeof(float) * nElem;  // Total bytes needed for each array

    // Allocate host (CPU) memory
    float *a_h = (float*)malloc(nByte);       // Host array a
    float *b_h = (float*)malloc(nByte);       // Host array b
    float *res_h = (float*)malloc(nByte);     // Host result from CPU computation
    float *res_from_gpu_h = (float*)malloc(nByte);  // Host result from GPU computation
    
    // Initialize result arrays to 0
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    // Allocate device (GPU) memory
    float *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((float**)&a_d, nByte));    // Device array a
    CHECK(cudaMalloc((float**)&b_d, nByte));    // Device array b
    CHECK(cudaMalloc((float**)&res_d, nByte));  // Device result array

    // Initialize input data on host
    initialData(a_h, nElem);  // Fill array a with initial values
    initialData(b_h, nElem);  // Fill array b with initial values

    // Copy data from host -> device
    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    // Define CUDA kernel execution configuration
    dim3 block(1024);  // Each block has 1024 threads
    dim3 grid(nElem / block.x);  // Calculate number of blocks needed
    
    // Launch the kernel to perform vector addition on GPU
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
    printf("Execution configuration<<<%d, %d>>>\n", grid.x, block.x);

    // Copy results back from device to host
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    
    // Perform the same computation on CPU for verification
    sumArrays(a_h, b_h, res_h, nElem);

    // Compare GPU and CPU results
    checkResult(res_h, res_from_gpu_h, nElem);

    // Free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    // Free host memory
    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}
