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

#srun python3 run_build_fitsdata.py /scratch/groups/kipac/alpv95/data/gen2a/ data/expanded/recon$SLURM_ARRAY_TASK_ID
#srun python3 run_build_fitsdata.py /scratch/groups/kipac/alpv95/data/GPD_EM_data/GPD_EM_6p4keV/ data/expanded/recon -meas
#srun python3 run_build_fitsdata.py /scratch/groups/rwr/alpv95/data/gen2a_5/ data/expanded/recon_5

#srun python3 run_build_fitsdata.py /home/users/alpv95/khome/tracksml/data/gen4_unpol /home/users/alpv95/khome/tracksml/data/expanded/newpaper_unpol --Erange 1.0 9.0 --fraction 0.0361 --pl 0 --aeff
#srun python3 run_build_fitsdata.py /home/users/alpv95/khome/tracksml/data/gen4_pol3 /home/users/alpv95/khome/tracksml/data/expanded/pol3 --Erange 1.0 9.0 --fraction 0.0091 --pl 0
srun python3 run_build_fitsdata.py /home/users/alpv95/khome/tracksml/data/spectra_68720/gen4_spec_true_flat_recon.fits /home/users/alpv95/khome/tracksml/data/spectra_68720/test_aug6 --tot 500000 --sim --augment 6




#python3 run_build_fitsdata.py /scratch/groups/rwr/alpv95/data/gen4_test data/expanded/final_6p4_unpol --augment 1 --shift 2
#srun python3 run_build_fitsdata.py /scratch/groups/rwr/alpv95/data/gen4_test/ data/expanded/ -meas
