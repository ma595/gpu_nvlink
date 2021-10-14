#include <stdio.h>
#include "nccl.h"
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
/* #include <constexpr> */
#include <stdexcept>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <type_traits>
#include "dot.cuh"

template <class T>
struct dependent_false : std::false_type
{
};

/* template <typename T> */
/* T dot_cublas(cublasHandle_t handle, std::size_t n, T* x, T* y) */
/* { */
/*   T result = 0; */
/*   if constexpr (std::is_same<T, double>()) */
/*     cublasDdot(handle, n, x, 1, y, 1, &result); */
/*   else if constexpr (std::is_same<T, float>()) */
/*     cublasSdot(handle, n, x, 1, y, 1, &result); */
/*   else */
/*     static_assert(dependent_false<T>::value); */

/*   return result; */
/* } */

template <typename T>
void check_arrays(int sz, T** vec);

// check arrays on device
template <typename T>
void check_arrays(int sz, T** vec){
  T* host_arr = (T*)malloc(sz * sizeof(T));
  /* cudaSetDevice(0); */
  T host_val = -999.;
  printf("norm val %f\n", host_val);
  cudaMemcpy(&host_val, &vec[0][0], sizeof(T), cudaMemcpyDeviceToHost);
  printf("host single value  : %f \n", host_val);
  cudaMemcpy(host_arr, vec[0], sz*sizeof(T), cudaMemcpyDeviceToHost);
  printf("host arr value  : %f \n", host_arr[0]);

  for (int i = 0; i < sz; i++){
    std::cout << host_arr[i] << std::endl;
  } 
}

// check cuda functions
template <typename T>
void test_init(){
  int nDev = 2;
  int sz = 32;

  T* host_v = (T*)malloc(sz * sizeof(T));
  for (int i = 0; i < sz; i++){
    host_v[i] = 10.0;
  }

  T** vec = (T**)malloc(nDev * sizeof(T*));

  T* devicebuff;
  /* for (int i = 0; i < nDev; ++i) { */
  /*   cudaSetDevice(i); */
  cudaMalloc(&devicebuff, sz * sizeof(T));
  cudaMemcpy((void*)devicebuff, (const void*)host_v, sz, cudaMemcpyHostToDevice); 

  T val = -999;
  cudaMemcpy(&val, &devicebuff[0], sizeof(T), cudaMemcpyDeviceToHost);
  printf("norm val %f\n", val);
  T* host_arr = (T*)malloc(sz * sizeof(T));
  cudaMemcpy(host_arr, devicebuff, sz*sizeof(T), cudaMemcpyDeviceToHost);
  printf("norm arr %f", host_arr[0]);
  /* } */
}

// reduce (sum) a vector over two devices.
template <typename T>
void reduce_sendbufftor(){
  int nDev = 2;
  int sz = 32;

  int devs[2] = {0, 1};
  T value = 10.;
  T* host_v = (T*)malloc(sz * sizeof(T));
  for (int i = 0; i < sz; i++){
    host_v[i] = value;
  }

  T** sendbuff = (T**)malloc(nDev * sizeof(T*));
  T** recvbuff = (T**)malloc(nDev * sizeof(T*));
  
  /* cudaMalloc(&recvbuff, nDev * sizeof(T)); */
  /* cudaMalloc(&reduced_result, nDev * sizeof(T)); */

  ncclComm_t comms[nDev];
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(i);
    cudaMalloc(sendbuff + i, sz * sizeof(T));
    cudaMalloc(recvbuff + i, sizeof(T));
    cudaMemcpy(sendbuff[i], host_v, sz * sizeof(T), cudaMemcpyHostToDevice); /* T float_val = 1.0; */ 
    cudaStreamCreate(s+i);
  }

  check_arrays(sz, sendbuff);
  ncclCommInitAll(comms, nDev, devs);
  ncclGroupStart();
  for (int i = 0; i < nDev; ++i)
    ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], sz, ncclFloat, ncclSum, comms[i], s[i]);
  ncclGroupEnd();
  
  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(s[i]);
  }

  T val = -999.;
  T* mem = (T*)malloc(nDev * sizeof(T));
  cudaMemcpy(&val, &recvbuff[0][0], sizeof(T), cudaMemcpyDeviceToHost);
  printf("single value %f\n", val);
}


// based on example 1: single_process_multiple_devices_nccl.c
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
//  Single Process, Single Thread, Multiple Devices
template <typename T>
int vector_norm_ssm(){
  int nDev = 2;
  int devs[2] = {0, 1};
  // vector length
  int size = 32;

  // initialize array on host
  T value = 10.;
  T* host_v = (T*)malloc(size * sizeof(T));
  for (int i = 0; i < size; i++){
    host_v[i] = value;
  }

  //allocating and initializing device buffers
  T** vec = (T**)malloc(nDev * sizeof(T*));
  T* dot_result = (T*)malloc(nDev * sizeof(T));
  T* reduced_result = (T*)malloc(nDev * sizeof(T));

  cudaMalloc(&reduced_result, nDev * sizeof(T));
  cudaMalloc(&dot_result, nDev * sizeof(T));

  ncclComm_t comms[nDev];
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  
  for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(i);
    cudaMalloc(vec + i, size * sizeof(T));
    cudaMemcpy(vec[i], host_v, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaStreamCreate(s+i);
  }
  check_arrays(size, vec);

  
  //initializing NCCL
  ncclCommInitAll(comms, nDev, devs);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE); // set here!!!

  
  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread

  T* result = (T*)malloc(sizeof(T));

  ncclGroupStart();
  for (int i = 0; i < nDev; ++i){
    // dot product kernel
    /* dot(handle, size * sizeof(T), vec[i], vec[i]); */
    /* dot<<<1,16>>>(size,  vec[i], vec[i], &recvbuff[i][0]); */
    /* dot<<<1,1>>>(size,  vec[i], vec[i], &dot_result[i]); */
    /* full_dot<<<1,1>>>(vec[i], vec[i], &dot_result[i], size); */

    // ncclAllReduce(vec, recvbuff, num_of_elements, type, reduction_operation, comms, streams)

    cublasSdot(handle, size, vec[i], 1, vec[i], 1, &dot_result[i]);
    /* cublasSdot(handle, size, vec[i], 1, vec[i], 1, result); */
    ncclAllReduce((const void*)&dot_result[i], (void*)&reduced_result[i], 1, ncclFloat,\
        ncclSum, comms[i], s[i]);
  }
  ncclGroupEnd();


 
  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(s[i]);
  }


  // check the value of the AllReduce  
  T val = -999.;
  cudaMemcpy(&val, &reduced_result[1], sizeof(T), cudaMemcpyDeviceToHost);
  printf("dot %f \n", val);
  printf("%f\n", result);
 

  cublasDestroy(handle);
  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(i);
    cudaFree(vec[i]);
    cudaFree(&dot_result[i]);
    cudaFree(&reduced_result[i]);
    /* cudaFree(recvbuff[i]); */
  }


  //finalizing NCCL
  for (int i = 0; i < nDev; ++i) 
    ncclCommDestroy(comms[i]);
  
  return 0;
}

int main()
{
  vector_norm_ssm<float>();
}
