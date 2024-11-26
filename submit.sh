#!/bin/bash
#
#SBATCH --job-name=GPUtest
#SBATCH --time=80:00
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=owners
#SBATCH --gres gpu:2
#SBATCH -C GPU_MEM:16GB
##SBATCH -C GPU_BRD:GEFORCE
##SBATCH -C GPU_SKU:RTX_2080Ti
##SBATCH -C GPU_CC:7.5

N="9"
DATA="09_det1"

nvidia-smi
python3 run_ensemble_eval.py $DATA --data_list data/mrk421/"$DATA"/ --batch_size 512
