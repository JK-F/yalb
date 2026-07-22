#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH -J lbm-gpu
#SBATCH --mem=90gb
#SBATCH --export=ALL
#SBATCH --partition=gpu_4

# Defaults for bwUniCluster's gpu_4 partition (4x NVIDIA V100 per node).
# Check these against your actual allocation before submitting:
#   sinfo -p gpu_4                                 # confirm the partition exists / is open to you
#   module avail cuda                              # confirm the exact CUDA module name/version below
#   sacctmgr show assoc user=$USER format=account,partition
module load compiler/gnu mpi/openmpi devel/cuda/11.8

echo "Running on ${SLURM_JOB_NUM_NODES} node(s) with ${SLURM_GPUS_ON_NODE:-4} GPU(s) per node."
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."

N=10000
SIZE=3000 # must divide evenly by each Cartesian dim: NP=4 -> dims=[2,2] -> 600x600 per GPU
lidv=0.03
omega=1.0
NP=${SLURM_NTASKS}

time mpirun -np $NP ./build-cuda/executables/milestone06 --N=$N --size_x=$SIZE --size_y=$SIZE --lidv=$lidv --omega=$omega

# Plotting is intentionally left out of this job: it needs matplotlib/ffmpeg
# (often missing on compute nodes) and doesn't need GPU allocation. Once the
# job finishes, run these on the login node:
#   python3 ./data/combine_csvs.py --output ./data/06_velocity.csv
#   python3 ./data/moving_lid.py $lidv $omega ./data/06_velocity.csv
