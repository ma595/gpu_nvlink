#include <stdio.h>
#include "nccl.h"
#include <stdlib.h>
/* #include <iostream> */
#include <stdexcept>
#include <type_traits>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "dot.cuh"

template <typename T>
void check_arrays(int sz, T** sendbuff);

template <typename T>
void check_arrays(int sz, T** sendbuff){
  T* host_arr = (T*)malloc(sz * sizeof(T));
  /* cudaSetDevice(0); */
  T host_val = -999.;
  printf("norm val %f\n", host_val);
  cudaMemcpy(&host_val, &sendbuff[0][0], sizeof(T), cudaMemcpyDeviceToHost);
  printf("host single value  : %f \n", host_val);
  cudaMemcpy(host_arr, sendbuff[0], sz*sizeof(T), cudaMemcpyDeviceToHost);
  printf("host arr value  : %f \n", host_arr[0]);

  /* for (int i = 0; i < sz; i++){ */
  /*   std::cout << host_arr[i] << std::endl; */
  /* } */ 
}

template <typename T>
void test_init(){
  int nDev = 2;
  int sz = 32;

  T* host_v = (T*)malloc(sz * sizeof(T));
  for (int i = 0; i < sz; i++){
    host_v[i] = 10.0;
  }

  T** sendbuff = (T**)malloc(nDev * sizeof(T*));

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

// based on example 1:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
// Example 1: Single Process, Single Thread, Multiple Devices
template <typename T>
void vector_norm_ssm(){
  int nDev = 2;
  int sz = 32;

  int devs[2] = {0, 1};
  /* cublasHandle_t handle; */
  /* cublasCreate(&handle); */
  T value = 10.;
  T* host_v = (T*)malloc(sz * sizeof(T));
  for (int i = 0; i < sz; i++){
    host_v[i] = value;
  }

  T** sendbuff = (T**)malloc(nDev * sizeof(T*));
  T* recvbuff;
  T* valbuff;
  
  cudaMalloc(&recvbuff, nDev * sizeof(T));
  cudaMalloc(&valbuff, nDev * sizeof(T));

  ncclComm_t comms[nDev];
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(i);
    cudaMalloc(sendbuff + i, sz * sizeof(T));
    cudaMemcpy(sendbuff[i], host_v, sz * sizeof(T), cudaMemcpyHostToDevice); /* T float_val = 1.0; */ 
    cudaStreamCreate(s+i);
  }

  check_arrays(sz, sendbuff);

  ncclCommInitAll(comms, nDev, devs);

  ncclGroupStart();
  /* cublasHandle_t handle; */
  /* cublasCreate(&handle); */
  for (int i = 0; i < nDev; ++i){
    /* T value = dot(handle, sz * sizeof(T), sendbuff[i], sendbuff[i]); */
    dot<<<1,32>>>(sz,  sendbuff[i], sendbuff[i], recvbuff);
    ncclAllReduce(recvbuff, valbuff, sz * sizeof(T), ncclFloat, ncclSum, comms[i], s[i]);
  }
  ncclGroupEnd();

  T val = -999.;
  cudaMemcpy(&val, &valbuff[0], sizeof(T), cudaMemcpyDeviceToHost);
  printf("single value %f", val);

}



int main()
{
  test_init_2<float>();

}
