#!/bin/bash

## Setup
OBS='9'
ATTNUM='4'
PREFIX='/home/users/alpv95/khome/tracksml/'
DATA_FOLDER=$PREFIX'moments/ixpeobssimdata/mrk421/01003'$OBS'01/'
DET='3'
INFILE='event_l1/ixpe01003'$OBS'01_det'$DET'_evt1_v01'

## Filter useless events.
ftcopy $DATA_FOLDER"$INFILE"'.fits[EVENTS][STATUS2 == b0x0000000000x00x]' $DATA_FOLDER"$INFILE"_filter.fits clobber=True
INFILE=$INFILE'_filter'
ixpeevtrecon infile=$DATA_FOLDER"$INFILE".fits outfile=$DATA_FOLDER"$INFILE"_recon.fits clobber=True writeTracks=True

## Add some missing header values.
fdump $DATA_FOLDER"$INFILE".fits[1] tmp.lis - 1 prdata=yes showcol=no
grep -i S_VDRIFT tmp.lis >> fix.lis
grep -i S_VBOT tmp.lis >> fix.lis
grep -I S_VGEM tmp.lis >> fix.lis
echo "FILE_LVL = '1'" >> fix.lis

fthedit $DATA_FOLDER"$INFILE"_recon.fits @fix.lis
rm tmp.lis fix.lis

############################################
## Now run NN analysis on _recon.fits...
## The results will be used to make _recon_nn.fits below.
############################################

ixpegaincorrtemp infile=$DATA_FOLDER"$INFILE"_recon.fits outfile=$DATA_FOLDER"$INFILE"_recon_gain.fits hkfile="$DATA_FOLDER"hk/ixpe01003"$OBS"01_all_pay_132"$DET"_v01.fits clobber=True logfile=recon.log
ixpechrgcorr infile=$DATA_FOLDER"$INFILE"_recon_gain.fits outfile=$DATA_FOLDER"$INFILE"_recon_gain_corr.fits initmapfile="$PREFIX"moments/CALDB/data/ixpe/gpd/bcf/chrgmap/ixpe_d"$DET"_20170101_chrgmap_01.fits outmapfile=$DATA_FOLDER"$INFILE"_chrgmap.fits phamax=60000.0 clobber=True
ixpegaincorrpkmap infile=$DATA_FOLDER"$INFILE"_recon_gain_corr.fits outfile=$DATA_FOLDER"$INFILE"_recon_gain_corr_map.fits clobber=True

## Replace MOM angles for NN angles here, and add NN weights, p_tail and flags.
## Need to have run the NN analysis on _recon before this step.
ftpaste $DATA_FOLDER"$INFILE"'_recon_gain_corr_map.fits[EVENTS][col -DETPHI2;]' $PREFIX'data_mrk421_0'$OBS'_det'$DET'___0'$OBS'_det'$DET'__ensemble.fits[1][col NN_PHI, DETPHI2==NN_PHI; NN_WEIGHT, W_NN==NN_WEIGHT; P_TAIL; FLAG]' $DATA_FOLDER"$INFILE"_recon_nn.fits history=YES clobber=True

ixpecalcstokes infile=$DATA_FOLDER"$INFILE"_recon_nn.fits outfile=$DATA_FOLDER"$INFILE"_recon_nn_stokes.fits clobber=True
## Use nn spmod files for spurious modulation correction.
ixpeadjmod infile=$DATA_FOLDER"$INFILE"_recon_nn_stokes.fits outfile=$DATA_FOLDER"$INFILE"_recon_nn_stokes_adj.fits clobber=True spmodfile="$PREFIX"moments/CALDB/data/ixpe/gpd/bcf/spmod/ixpe_d"$DET"_20170101_spmod_nn.fits
ixpeweights infile=$DATA_FOLDER"$INFILE"_recon_nn_stokes_adj.fits outfile=$DATA_FOLDER"$INFILE"_recon_nn_stokes_adj_w.fits clobber=True

## Add some missing header values.
yes | ftlist $DATA_FOLDER"$INFILE".fits["EVENTS"] outfile=STDOUT ROWS=0 | grep TC > wcs.lis
sed -i 's/36 /44 /;s/37 /45 /' wcs.lis
echo -e "TLMIN44 = 1\nTLMAX44 = 600\nTLMIN45 = 1\nTLMAX45 = 600" >> wcs.lis
fthedit $DATA_FOLDER"$INFILE"_recon_nn_stokes_adj_w.fits["EVENTS"] @wcs.lis

## Coordinate transformations.
ixpedet2j2000 infile=$DATA_FOLDER"$INFILE"_recon_nn_stokes_adj_w.fits outfile=$DATA_FOLDER"$INFILE"_recon_nn_stokes_adj_w_j2000.fits attitude="$DATA_FOLDER"hk/ixpe01003"$OBS"01_det"$DET"_att_v0"$ATTNUM".fits clobber=True
ixpeaspcorr infile=$DATA_FOLDER"$INFILE"_recon_nn_stokes_adj_w_j2000.fits clobber=True n=300 att_path="$DATA_FOLDER"hk/ixpe01003"$OBS"01_det"$DET"_att_v0"$ATTNUM".fits