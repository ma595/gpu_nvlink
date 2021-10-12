__device__ unsigned int count = 0;

template <typename T, int BLOCK_SIZE = 512>
__global__ void dot(int n, T* x, T* y, T* result)
{
  __shared__ T cache[BLOCK_SIZE];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  cache[threadIdx.x] = 0.;
  while (index < n)
  {
    cache[threadIdx.x] += x[index] * y[index];
    index += stride;
  }
  __syncthreads();
  index = BLOCK_SIZE / 2;
  while (index > 0)
  {
    if (threadIdx.x < index)
      cache[threadIdx.x] += cache[threadIdx.x + index];
    __syncthreads();
    index /= 2;
  }
  bool last_block = false;
  if (threadIdx.x == 0)
  {
    result[blockIdx.x] = cache[0];
    __threadfence();
    const unsigned int v = atomicInc(&count, gridDim.x);
    last_block = (v == gridDim.x - 1);
  }

  if (last_block && threadIdx.x == 0)
  {
    float r = 0;
    for (float* b = result; b != result + gridDim.x; r += *b++)
      result[0] = r;
  }

  
}


typedef float real_t;

const size_t BLOCK_SIZE = 16;

__global__ void full_dot(const real_t* v1, const real_t* v2, real_t* out,
                         int N) {
    __shared__ real_t cache[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[threadIdx.x] = 0.f;
    while (i < N) {
        cache[threadIdx.x] += v1[i] * v2[i];
        i += gridDim.x * blockDim.x;
    }
    __syncthreads();  // required because later on the current thread is
                      // accessing data written by another thread
    i = BLOCK_SIZE / 2;
    while (i > 0) {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
        i /= 2;  // not sure bitwise operations are actually faster
    }
#ifndef NO_SYNC  // serialized access to shared data;
    if (threadIdx.x == 0) atomicAdd(out, cache[0]);
#else  // no sync, what most likely happens is:
       // 1) all threads read 0
       // 2) all threads write concurrently 16 (local block dot product)
    if (threadIdx.x == 0) *out += cache[0];
#endif
}
