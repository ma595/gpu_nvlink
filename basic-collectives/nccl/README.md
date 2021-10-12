# NCCL experiments

NCCL documentation with some useful examples is found [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)

There are two examples:
- Example 1 [vector_norm_nccl](src/vector_norm_nccl.cu), which has one thread or process and multiple devices. A communicator object is created for each device. 
- Example 2 [vector_norm_nccl-mpi](src/vector_norm_nccl-mpi.cu) which has one process / host thread responsible for at most one GPU. 

Example 2 is executed with MPI:

``` 
mpirun -np 2 ./mpi
```

In both examples a cuda kernel does a vector dot product (src/dot.cuh) and a ncclAllReduce sums the result over all devices. 

