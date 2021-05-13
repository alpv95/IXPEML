#!/bin/bash
#
#SBATCH --job-name=GPUtest
#SBATCH --time=150:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --partition=owners
#SBATCH --gres gpu:2
####SBATCH -C GPU_SKU:TESLA_P100_PCIE
#SBATCH -C GPU_MEM:16GB
#SBATCH -C GPU_BRD:GEFORCE

ml python/3.6.1
ml py-scipy/1.1.0_py36
ml viz
ml py-matplotlib/2.1.2_py36
ml py-numpy/1.17.2_py36
ml cuda/11.2.0
ml cudnn/7.6.5

nvidia-smi
srun python3 run_ensemble_eval.py energy3 --data_list spectra2/ISP2/train/ --ensemble energy --datatype sim
