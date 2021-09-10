#! /bin/bash 
#SBATCH --nodes=1
#SBATCH -p pascal
#SBATCH --account SUPPORT-GPU
#SBATCH --exclusive
#SBATCH -t 0:02:00
#SBATCH --output "out/pascal_%j.out"

. /etc/profile.d/modules.sh
module purge
export OMP_NUM_THREADS=1
module load slurm
module add cuda/8.0 gcc-5.4.0-gcc-4.8.5-fis24gg openmpi-1.10.7-gcc-5.4.0-jdc7f4f
nvcc examples/p2pBandwidthLatencyTest.cu -I/usr/local/software/cuda/11.4/samples/common/inc/ -o pascal
./pascal --p2p_read
