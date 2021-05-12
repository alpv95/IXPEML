# Track Reconstruction with Machine Learning

This package contains machine learning-based track reconstruction code for the Imaging X-ray Polarimetry Explorer (IXPE). This is v1.3 of the project, there will be updates.

## Collaborating
* When pulling/forking the repository, note that `net_archive/` takes up ~1GB of storage due to the large neural network (NN) parameter files.
* Please make your own branch and do not push to master.
* Let me know of any difficulties, feature requests etc. at alpv95@stanford.edu

## Setup

There are two separate packages involved here: Deep Learning track reconstruction & GPDSW.
The deep learning track reconstruction package runs python 3.6.1 -- gpdsw and gpdext (found in moments/gpd...) run python 2.7.1.

The setup for GPDSW can be found on the IXPE wiki, with the original code repository found on the [IXPE bitbucket](https://bitbucket.org/ixpesw/workspace/projects/IGS). GPDSW is not provided here and must be installed independently by the user. Note: it is essential to have the most up to date GPDSW version.
For the Deep Learning track reconstruction, the required packages are summarized in ```requirements.txt```.

## Running

See the example jupyter notebook ```example/Example.ipynb```. An example calibration dataset is included in a google drive (~0.5GB total), and can be downloaded and saved in ```example/data/```. We give a brief summary of the pipeline below.



Using the Deep Learning (DL) track reconstruction is a 3-step process from a raw fits file of GPD detector data. There is also the option to generate your own simulated data (step 0). Step 4 is the post-processing, this is up to the user so we just describe the final results format and how to interpret it.
Our pipeline additionally runs the standard moments analysis in parallel for comparison. 

0. Use GPDSW to generate arbitrary simulated data. The data must have gem effective gain set to 190, as shown below. We trained the NNs on a DME partial pressure of 733mbar. We recommend simulating data at 733mbar since this corresponds to true pressure ~688mbar (the simulation software was miscalibrated). 


1. Run the GPDSW track reconstruction analysis on the desired raw fits file. This applies the moment analysis and extracts
individual hexagonal tracks into a new fits file with new name "..._recon.fits". For the current DL settings the charge threshold=10 (this is what the NNs were trained for); the "write-tracks" option must also be activated.

2. Convert the hexagonal tracks saved in the "..._recon.fits" files (step 1) into square track datasets (with labels - photoelectron angles, absorption points, energies - if the tracks are simulated) so that the NNs can take them as input. These
are saved as pytorch `.pt` files. This is done using `run_build_fitsdata.py`, for example for real detector tracks:
```
python3 run_build_fitsdata.py /input/recon_fits_file /output/directory --meas 
```
and for simulated tracks
```
python3 run_build_fitsdata.py /input/recon_fits_file /output/directory --tot 50000
```

3. Run the DL ensemble track reconstruction on the dataset(s) formed in step 2. This is done by running `run_ensemble_eval.py`.
All of the results for individual tracks (including the moment analysis results) are saved in a fits file at the end of this run. Example code:
```
python3 run_ensemble_eval.py fits_filename_where_results_will_be_saved --data_list example_dataset/train/ 
```
This requires a GPU to be available for reasonable runtime (ideally multiple if the dataset is very large, >~ 1e6 tracks).

## Post-processing

4. The final outputs in the saved pickle files are listed in order (each is an element of the tuple):

ColDefs(
* name = 'NN_PHI'; format = 'E'  -- NN predicted photoelectron angles
* name = 'MOM_PHI'; format = 'E' -- Moment analysis predicted photoelectron angles
* name = 'PHI'; format = 'E' -- True photoelectron angles (if data is simulated, otherwise this is None)
* name = 'MOM_ELLIP'; format = 'E' -- Moment analysis measured track ellipticies
* name = 'NN_WEIGHT'; format = 'E' -- NN predicted weights for photoelectron angles
* name = 'NN_ABS'; format = '2E'; dim = '(2)' -- NN predicted absorption points (xy coords on the square track image grid 50x50)
* name = 'MOM_ABS'; format = '2E'; dim = '(2)' Moments predicted absorption points (xy coords on the square track image grid 50x50)
* name = 'XY_MOM_ABS'; format = '2E'; dim = '(2)' -- Moments predicted absorption points (xy coords on the detector grid)
* name = 'ABS'; format = '2E'; dim = '(3)' -- True absorption points (xy coords on the square track image grid 50x50) (if data is simulated, otherwise this is None)
* name = 'ENERGY'; format = 'E' -- True track energies in keV (if data is simulated, otherwise this is None)
* name = 'NN_ENERGY'; format = 'E' -- NN predicted track energies in keV
* name = 'XY_NN_ABS'; format = '2E'; dim = '(2)' -- NN predicted absorption points (xy coords on the detector grid)
* name = 'XYZ_ABS'; format = '3E'; dim = '(3)' -- True absorption points (xyz coords on the detector grid) (if data is simulated, otherwise this is None)
* name = 'MOM_ENERGY'; format = 'E' -- MOM predicted track energies in keV
* name = 'NN_WEIGHT_EPIS'; format = 'E' -- NN predicted weights (epistemic) for photoelectron angles
)

Example applications are shown the Example/example.ipynb.


