#include <stdio.h>
#include "nccl.h"
#include <stdlib.h>
#include "mpi.h"
/* #include <iostream> */
#include <stdexcept>
/* #include <type_traits> */
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "dot.cuh"
#define N (64)
#define THREADS_PER_BLOCK 512

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

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


/* static uint64_t getHostHash(const char* string) { */
/*   // Based on DJB2a, result = result * 33 ^ char */
/*   uint64_t result = 5381; */
/*   for (int c = 0; string[c] != '\0'; c++){ */
/*     result = ((result << 5) + result) ^ string[c]; */
/*   } */
/*   return result; */
/* } */

template <typename T>
void check_arrays(int sz, T** sendbuff);


// simple MPI test function
template <typename T>
void test_MPI(){
  printf("Hello world");
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
  int myRank, nRanks, localRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  printf("%d", myRank);

}


// assign values to two vectors and reduce.
void test_vector_reduce(){
  int myRank, nRanks, localRank = 0;

  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  int sz = 32; 
  int size = sz;
  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
 
  float value = 10.;
  float* host_v = (float*)malloc(sz * sizeof(float));
  for (int i = 0; i < sz; i++){
    host_v[i] = value;
  }
  cudaMemcpy(sendbuff, host_v, sz * sizeof(float), cudaMemcpyHostToDevice); /* T float_val = 1.0; */ 

  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));


  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));


  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  float val = -999.;
  cudaMemcpy(&val, &recvbuff[0], sizeof(float), cudaMemcpyDeviceToHost);
  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);


  MPI_Finalize();
  printf("single value %f\n", val);

}

/* template <class T> */
/* struct dependent_false : std::false_type */
/* { */
/* }; */

template <typename T>
void check_arrays(int sz, T* sendbuff){
  T* host_arr = (T*)malloc(sz * sizeof(T));
  /* cudaSetDevice(0); */
  T host_val = -999.;
  printf("norm val %f\n", host_val);
  cudaMemcpy(&host_val, &sendbuff[0], sizeof(T), cudaMemcpyDeviceToHost);
  printf("host single value  : %f \n", host_val);
  cudaMemcpy(host_arr, sendbuff, sz*sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++){
    printf("host arr value  : %f \n", host_arr[i]);
  }

  /* for (int i = 0; i < sz; i++){ */
  /*   std::cout << host_arr[i] << std::endl; */
  /* } */ 
}

template <typename T>
void reduce_vector(){
  int myRank, nRanks, localRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  /* printf("%d", myRank); */

  ncclUniqueId id;
  ncclComm_t comm;
  T *sendbuff, *recvbuff;
  cudaStream_t s;
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // this is the initial size of the vector - we reduce on each device using a kernel
  int sz = 32; 
  int size = sz;
  T* sendbuff_host = (T*)malloc(sz * sizeof(T));
  T* recvbuff_host = (T*)malloc(sz * sizeof(T));
  
  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  cudaMalloc(&sendbuff, size * sizeof(T));
  cudaMalloc(&recvbuff, size * sizeof(T));

  // assign 10's to vector which we reduce on each device
  T value = 10.;
  T* host_v = (T*)malloc(sz * sizeof(T));
  for (int i = 0; i < sz; i++){
    host_v[i] = value;
  }
  cudaMemcpy((void*)sendbuff, (const void*)host_v, sz * sizeof(T), cudaMemcpyHostToDevice); /* T T_val = 1.0; */ 

  /* CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(T))); */
  CUDACHECK(cudaStreamCreate(&s));

  /* check_arrays<T>(sz, sendbuff); */
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  T *a;
  a = (T*)malloc(sizeof(T));
  T *dev_a;
  cudaMalloc((void **)&dev_a, sizeof(T));
  *a = 0;

  /* cudaMemcpy(dev_a, a, sizeof(T), cudaMemcpyHostToDevice); */

    /* dot<<<2, 512>>>(size,  sendbuff, sendbuff, dev_a); */
  full_dot<<<1,32>>>(sendbuff, sendbuff, dev_a, sz);
  /* cudaDeviceSynchronize(); */

  // check cudamemcpy 
  cudaMemcpy(a, dev_a, sizeof(T), cudaMemcpyDeviceToHost);

  printf("dot %f\n", *a);

  T *dev_b;
  cudaMalloc((void **)&dev_b, sizeof(T));
  
  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)dev_a, (void*)dev_b, 1, ncclFloat, ncclSum, comm, s));


  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  float val = -999.;
  cudaMemcpy(&val, &dev_b[0], sizeof(float), cudaMemcpyDeviceToHost);
  printf("dot %f \n", val);
  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  //finalizing NCCL
  ncclCommDestroy(comm);

}

int main(int argc, char* argv[])
{
  /* vector_norm_ssm<float>(); */
  MPI_Init(&argc, &argv);
  /* /1* test_mpi<float>(); *1/ */
  reduce_vector<float>();
  /* printf("Hello world"); */
  MPI_Finalize();
  /* printf("single value %f\n", val); */

}
