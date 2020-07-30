# Track Reconstruction with Machine Learning

This package contains machine learning-based track reconstruction code for the Imaging X-ray Polarimetry Explorer (IXPE). This is v1.2 of the project, there will be updates.

## Collaborating
* When pulling/forking the repository, note that `net_archive/` takes up ~1GB of storage due to the large neural network (NN) parameter files.
* Please make your own branch and do not push to master.
* Let me know of any difficulties, feature requests etc. at alpv95@stanford.edu

## Setup

There are two separate packages involved here: Deep Learning track reconstruction & GPDSW.
The deep learning track reconstruction package runs python 3.6.1 -- gpdsw and gpdext (found in moments/gpd...) run python 2.7.1.

The setup for GPDSW can be found on the IXPE wiki, with the original code repository found on the [IXPE bitbucket](https://bitbucket.org/ixpesw/workspace/projects/IGS). There are some very minor additions to the original GPDSW here (in the Io/), to let full hexagonal tracks and their moment analysis results be properly saved. GPDSW is not provided here and must be installed independently by the user. Note: it is essential to have GPDSW version >9.0.
For the Deep Learning track reconstruction, the required packages are summarized in ```requirements.txt```.

## Running

See the example jupyter notebook ```Example/example.ipynb```. Example datasets are included in a google drive (~1.5GB total), and should be downloaded and saved in ```Example/data/```. We give a brief summary of the pipeline below.



Using the Deep Learning (DL) track reconstruction is a 3-step process from a raw fits file of GPD detector data. There is also the option to generate your own simulated data (step 0). Step 4 is the post-processing, this is up to the user so we just describe the final results format and how to interpret it.
Our pipeline additionally runs the standard moments analysis in parallel for comparison. 

0. Use GPDSW to generate arbitrary simulated data. The data must have gem effective gain set to 190, as shown below. We trained the NNs on a DME partial pressure of 733mbar. We recommend simulating data at 733mbar since this corresponds to true pressure ~688mbar (the simulation software was miscalibrated). 
For example in the `moments/gpdsw/bin` directory run:
```bash
./ixpesim -n 37000 --random-seed 131 --output-file /scratch/groups/rwr/alpv95/data/gen4_test/gen4_3p7_unpol.fits --log-file /scratch/groups/rwr/alpv95/data/gen4_test/gen4_3p7_unpol.log --src-energy 3.7 --src-polarized 0 --src-pol-angle 90 --gem-eff-gain 190 --dme-pressure 733
```
to generate 37000 unpolarized tracks at 3.7kev. 


1. Run the GPDSW track reconstruction analysis on the desired raw fits file. This applies the moment analysis and extracts
individual hexagonal tracks into a new fits file with new name "..._recon.fits". For the current DL settings the charge threshold=10 (this is what the NNs were trained for); the "write-tracks" option must also be activated. This can be done by running the following in the `moments/gpdsw/bin directory`:
```
./ixperecon --write-tracks --input-files ~/.../.../.../raw_data.fits --threshold 10 --output-folder ~/.../.../.../
```

2. Convert the hexagonal tracks saved in the "..._recon.fits" files (step 1) into square track datasets (with labels - photoelectron angles, absorption points, energies - if the tracks are simulated) so that the NNs can take them as input. These
are saved as pytorch `.pt` files. This is done using `run_build_fitsdata.py`, for example for real detector tracks:
```
python3 run_build_fitsdata.py /input/directory/ /output/directory --meas example_recon.fits --meas_e 3.7 --meas_tot 37000
```
and for simulated tracks
```
python3 run_build_fitsdata.py /input/directory/ /output/directory --Erange 1.8 8.2 --fraction 0.005 --pl 0 --shuffle
```
The output directory will split the initial hexaginal dataset into 3 folders: train, val and test. The fraction of total tracks in each of these can be edited in `run_build_fitsdata.py`.
Each individual hexagonal track gets 6 square conversions: two hex-to-square pixel shifts for each 120deg rotation. This allows us to later re-rotate and remove the hex-to-square biases. Each of the 6 track images is 30x30 pixels, so depending on the size of the dataset, the output file could be large.  

3. Run the DL ensemble track reconstruction on the dataset(s) formed in step 2. This is done by running gpu_test.py with the ensemble flag. There is currently on one choice of DL ensemble: "bigE".
All of the results for individual tracks (including the moment analysis results) are saved in a pickle file at the end of this run. The moments, moments_cuts, NN and weighted NN (lambda = 2) modulation/phi results are printed at the end. Example code:
```
python3 gpu_test.py --data_list example_dataset/train/ --save pickle_filename_where_results_will_be_saved.pickle --ensemble bigE
```
This requires a GPU to be available for reasonable runtime (ideally multiple if the dataset is very large, >~ 1e6 tracks).

Note: In each of the steps/files there are a number of options available to customize. These should be obvious at the top of each file in the arg_parse section.

## Post-processing

4. The final outputs in the saved pickle files are listed in order (each is an element of the tuple):

* angles_nn -- NN predicted photoelectron angles, numpy array of floats shape (N, 3, n) --> **N is the number of tracks**, 3 120deg rotations to remove hex2square bias, and **n the number of NNs in the ensemble (~10)**
* angles_mom -- Moment analysis predicted photoelectron angles, numpy array of floats shape (N) 
* angles_sim -- True photoelectron angles (if data is simulated, otherwise this is None), numpy array of floats shape (N) 
* ellipticities -- Moment analysis measured track ellipticies, numpy array of floats shape (N) 
* errors -- NN predicted uncertainties (sigmas) on photoelectron angles, numpy array of floats shape (N, 3, n)
* abs_pts_nn -- NN predicted absorption points (xy coords on the square track image grid 30x30), numpy array of floats shape (N, 2)
* mom_abs_pts -- Moments predicted absorption points (xy coords on the square track image grid 30x30), numpy array of floats shape (N, 2)
* abs_pts_sim -- True absorption points (xy coords on the square track image grid 30x30) (if data is simulated, otherwise this is None), numpy array of floats shape (N, 2)
* energies_nn -- NN predicted track energies in keV, numpy array of floats shape (N) 
* energies_sim -- True track energies in keV (if data is simulated, otherwise this is None), numpy array of floats shape (N)
* angles1 -- Diagnostic/testing Tool (ignore)
* errors1 -- Diagnostic/testing Tool (ignore)
* xy_abs_pts -- Moments predicted absorption points (xy coords on the detector grid), numpy array of floats shape (N, 2). These are to convert from abs_pts on the NN track image grid to detector abs_pts.

Example applications are shown the Example/example.ipynb.


