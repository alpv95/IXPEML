#!/bin/bash
#
#SBATCH --job-name=GPUtest
#SBATCH --time=150:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --partition=owners
#SBATCH --gres gpu:2

ml python/3.6.1
ml py-scipy/1.1.0_py36
#ml py-pytorch/1.0.0_py36
ml viz
ml py-matplotlib/2.1.2_py36
ml py-numpy/1.17.2_py36

#srun python3 gpu_test.py --data_list stokes/measgen4_2p7_None_unpol/train/ --stokes_correct 2.7 --ensemble pol_only
#srun python3 gpu_test.py --data_list review_unpol/train/ --ensemble flat --save flat_all_unpol

#srun python3 gpu_test.py --model_list 733flat1_mserrall2_expectile0p2_alpha0p8 --data_list review_unpol/test/ --save alphaflat2
srun python3 gpu_test.py --data_list pol_heads/train/ --ensemble bessel_rand --save pol_heads_bess

#srun python3 gpu_test.py --save review2_733unpol_test --data_list fom_unpol/test/ --model_list 733flat_mserrall1 733flat_mserrall1_3 733flat_mserrall1_El1loss_2 733flat_mserrall1_small1 733flat_mserrall1_small2 733_mserrall1_2 733flat_mserrall1_2 733flat_mserrall1_El1loss 733flat_mserrall1_El1loss_small 733flat_mserrall1_small1_2 733_mserrall1
#srun python3 gpu_test.py --data_list paper_plot/train/ --ensemble pol_abs_E --save energy_test
#srun python3 gpu_test.py --input_channels 2 --save final4_alpha03_absE --model_list final_MSERRALL1_PL1_aug1_alpha1 final_MSERRALL1_PL2_aug1_alpha1_Z final_MSERRALL1_PL2_aug1_alpha1_abs03_N1 final_MSERRALL1_PL1_aug1_alpha1_abs03_N1 final_MSERRALL1_PL2_aug1_alpha1_abs03 final_MSERRALL1_PL1_aug1_alpha1_abs03 final_MSERRALL1_PL2_aug1_alpha1_lowreg_abs03 final_MSERRALL1_PL2_aug1_alpha1_lowreg_abs03_N1 final_MSERRALL1_PL1_aug1_alpha1_lowreg_abs03 final_MSERRALL1_PL1_aug1_alpha1_lowreg_abs03_N1 final_MSERRALL1_PL1_aug1_alpha2_lowreg_abs03 final_MSERRALL1_PL2_aug1_alpha2_lowreg_abs03 final_MSERRALL1_PL1_aug2_alpha2_lowreg_abs005 final_MSERRALL1_PL2_aug2_alpha2_lowreg_abs005 --data_list final_2p0_unpol/train/ final_2p7_unpol/train/ final_3p7_unpol/train/ final_4p5_unpol/train/ final_5p4_unpol/train/ final_6p4_unpol/train/ final_7p5_unpol/train/ gen4_pl1_aug1_final/test/ gen4_pl2_aug1_final/test/ 
#srun python3 gpu_test.py --method weighted_MLE --save LOW3kev_mle --model_list gen4_MSERR2dropPL2LOW3kev_aug1_noshift gen4_MSERR2dropPL2LOW3kev_aug1_noshift_lowreg gen4_MSERR2dropPL2LOW3kev_aug1_shift gen4_MSERR2dropPL2LOW3kev_aug1_noshift_highreg gen4_MSERR2dropPL2LOW3kev_aug1_noshift_Z  --data_list simgen4_2p7_unpol/train/ #gen4_pl2LOW3kev_aug1_shift/test/ gen4_pl2LOW3kev_aug1_noshift/test/
#srun python3 gpu_test.py --save HIGH --model_list gen4_MSERR2dropPL2HIGH_aug1_noshift gen4_MSERR2dropPL2HIGHsmall_aug1_noshift gen4_MSERR2dropPL2HIGHsmall_aug1_noshift_N3 gen4_MSERR2dropPL2HIGHmedium_aug1_noshift gen4_MSERR2dropPL2HIGHsmall_aug1_noshift_N2 gen4_MSERR2dropPL2HIGHsmall_aug6_noshift --data_list simgen4_6p4_unpol/train/ gen4_pl2HIGH_aug1_noshift/test/

#srun python3 gpu_test.py resnet_reconERRdrop -d test/ --quality_cut 0
#srun python3 gpu_test.py resnet_reconERRdrop -d meas6p4_ROB/train/ --quality_cut 0
#srun python3 gpu_test.py resnet_reconERRdrop -d meas6p4_ROB/train/ --quality_cut 2
#srun python3 gpu_test.py resnet_reconERRdrop -d meas6p4_ROB/train/ --quality_cut 3
#srun python3 gpu_test.py resnet_reconERRdrop -d meas2p7_UnPol/train/ --quality_cut 0
#srun python3 gpu_test.py resnet_reconERRdrop -d meas2p7_UnPol/train/ --quality_cut 1.2
#srun python3 gpu_test.py resnet_reconERRdrop -d meas2p7_UnPol/train/ --quality_cut 1.5
