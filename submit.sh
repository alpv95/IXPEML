#!/bin/bash
#
#SBATCH --job-name=GPUtest
#SBATCH --time=180:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --partition=owners
#SBATCH --gres gpu:2

ml python/3.6.1
ml py-scipy/1.1.0_py36
ml viz
ml py-matplotlib/2.1.2_py36
ml py-numpy/1.17.2_py36

srun python3 gpu_test.py --data_list spectra/ISP/train/ --save bess3

