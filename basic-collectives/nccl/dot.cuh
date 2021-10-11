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


