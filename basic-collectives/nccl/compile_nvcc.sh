
module purge
export OMP_NUM_THREADS=1
module load slurm
module load cuda/8.0
module load nccl/2.4.2-cuda9.2
module load gcc-5.4.0-gcc-4.8.5-fis24gg
nvcc -std=c++11 vector_norm_nccl.cu -I/usr/local/software/nccl/2.4.2-cuda9.2/include -L//usr/local/software/nccl/2.4.2-cuda9.2/lib -lcudart -lnccl -lcuda -lcublas link.o -o vnnccl


# module load rhel8/default-amp
# module load cuda/11.4 openmpi-4.0.5-gcc-8.4.1-2lpgttf
# . ~/asimov/asimov-rds/spack-install/spack/share/spack/setup-env.sh
# module load openmpi-1.10.7-gcc-5.4.0-jdc7f4f
# module load gcc/9
