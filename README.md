# gpu-nvlink

## Querying nvlink functionality on A100s

``` 
gpu-q-8> nvidia-smi nvlink -c  

Link 0, P2P is supported: true
Link 0, Access to system memory supported: true
Link 0, P2P atomics supported: true
Link 0, System memory atomics supported: true
Link 0, SLI is supported: true
Link 0, Link is supported: false
Link 1, P2P is supported: true
Link 1, Access to system memory supported: true
Link 1, P2P atomics supported: true
Link 1, System memory atomics supported: true
Link 1, SLI is supported: true
Link 1, Link is supported: false
```

12 links (each 25GB/s), 600 GB/s total. See:

```
nvidia-smi nvlink -s
nvidia-smi topo --matrix
nvidia-smi topo -p2p n

GPU0   GPU1    GPU2    GPU3
GPU0   X       OK      OK      OK
GPU1   OK      X       OK      OK
GPU2   OK      OK      X       OK
GPU3   OK      OK      OK      X
``` 

How do we actually monitor/verify NVLink utility?

# Benchmarks

## NVIDIA CUDA Toolkit

The file [p2pBandwidthLatencyTest.cu](examples/p2pBandwidthLatencyTest.cu) demonstrates the CUDA Peer-To-Peer (P2P) data transfers between pairs of GPUs and computes latency and bandwidth. Tests on GPU pairs using P2P and without P2P. Output on A100:

```
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3
     0   2.87   2.63   2.63   2.72
     1   2.64   2.75   2.67   2.64
     2   2.81   2.80   3.08   2.81
     3   2.80   2.82   2.79   2.88
```

Output on p100:
```
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3
     0   1.83   1.33   1.38   1.34
     1   1.35   1.84   1.34   1.39
     2   1.35   1.36   1.85   1.31
     3   1.34   1.35   1.38   1.85
```

So performance appears to be better on p100???
What is this actually doing?

cudaMemcpy(gpu0_buf, gpu1_buf, buf_size, cudaMemcpyDefault)

## gdrcopy
A low-latency GPU memory copy library. 
[gdrcopy](https://github.com/NVIDIA/gdrcopy) provides `copylat` and `copybw`. Can either install from spack or by cloning the repo and running make. 


## ceed benchmarks

[CEED](https://github.com/CEED/benchmarks)   
[EXCALIBUR-CEED](https://github.com/Excalibur-SLE/ceed-benchmarks/)


# Notes

nvidia-smi top -m

spack info /rkwciv5  
/usr/local/software/spack/a100-hpl  
cudamemcpy  
module load gdrcopy  
MPS  
USM   


not between all devices  
buffers and accessors  
breaking previous model of scheduling of kernels.  
nvidia-smi nvlink -s

https://www.youtube.com/watch?v=EX-xSb1QAT4
> https://stackoverflow.com/questions/53174224/nvlink-or-pcie-how-to-specify-the-interconnect  
> https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c
