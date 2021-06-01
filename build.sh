#!/bin/bash
#
#SBATCH --job-name=BUILD
#SBATCH --time=120:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=25G
#SBATCH --partition=kipac
##SBATCH --array=5-8 ##these are the SLURM task ids

ml python/3.9
ml py-scipy/1.6.3_py39
ml viz
ml py-matplotlib/3.4.2_py39
ml py-numpy/1.20.3_py39

srun python3 run_build_fitsdata.py /home/users/alpv95/khome/tracksml/data/spectra_68720/gen4_spec_true_flat_recon.fits /home/users/alpv95/khome/tracksml/data/spectra_68720/test_aug6 --tot 500000 --sim --augment 6
