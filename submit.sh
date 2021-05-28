#!/bin/bash
#
#SBATCH --job-name=GPUtest
#SBATCH --time=120:00
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --partition=owners
#SBATCH --gres gpu:1
#SBATCH -C GPU_MEM:16GB
#SBATCH -C GPU_BRD:GEFORCE
#SBATCH -C GPU_SKU:RTX_2080Ti
#SBATCH -C GPU_CC:7.5

ml gsl
ml python/3.9
ml py-scipy/1.6.3_py39
ml viz
ml py-matplotlib/3.4.2_py39
ml cmake/3.13.1
ml py-numpy/1.20.3_py39
ml cudnn/8.1.1.33
ml py-pytorch/1.8.1_py39
ml cuda/11.1.1

N="9"

nvidia-smi
srun python3 run_ensemble_eval.py heads_only_choice --data_list spectra_68720/test_aug6/train/ --ensemble heads_only_new --datatype sim --batch_size 512
