# Mass deploy convolutional net
# TODO: Consider saving per-multistart results in separate files (so any multistart fail doesn't spread)
#   Or this can be ignored when doing 1 multistart per job for more train data jobs.
import os
import sys
import shutil
import itertools
import copy
import torch
import numpy as np
sys.path.insert(0, '/home/groups/rwr/alpv95/tracksml')
from util import h5pack
from util.submit_cluster import submit as submit_cluster
from util.pydataloader import H5Dataset, ToTensor, ZNormalize, SelfNormalize
from collections import namedtuple
from nn.cnn import TrackAngleRegressor
from formats.dense_square import DenseSquareSimTracks
from torchvision import transforms


def main():
    Spec = namedtuple('Spec', ['name', 'data_file', 'loss', 'alpha_loss','Z','lambda_abs','lambda_E','optim_method',
                        'input_channels','n_multistarts', 'n_multistarts_per_job','n_threads', 'subset'])

    use_cluster = True
    data_dir = os.path.realpath('data/')
    data_file = data_dir + "/expanded/train_heads_low/"

    job = Spec(name='733flat_mserrall2_headonly_low', data_file=data_file, loss='mserrall2', alpha_loss=0.8, Z=None, lambda_abs=0.2, lambda_E=0.2,
                optim_method='RLRP', input_channels=2, n_multistarts=1, n_multistarts_per_job=1, n_threads=1, subset=False)
    
    # Set dirs
    job_dir = os.path.join(data_dir, 'nn/', job.name)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # Copy data to job dir
    shutil.copy2(data_file + "train/ZN.pt", job_dir)
    shutil.copy2(data_file + "train/ZNE.pt", job_dir)

    # Hyperparams
    layer_sizes = [128, 64]  # for the 1st 2 fully connected layers; the last has 1 or 2 depending on loss fun
    layer_combos = list(itertools.product(layer_sizes, repeat=2))
    # Make sure all architectures are "pyramidal"
    layer_combos_keep = [layer for layer in layer_combos if check_pyramidal(layer)]

    hparams = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.0095, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 4096, #[64],
        'verbose': 1, #[1],
        'optim_method': job.optim_method,
        'dropout': 0.0,
        'alpha_loss': job.alpha_loss,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'Z': job.Z,
        'input_channels': job.input_channels,
        'n_epochs': 191, #[50]
    } 
    hparams1 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.0095, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 2048, #[64],
        'verbose': 1, #[1],
        'optim_method': job.optim_method,
        'Z': job.Z,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams2 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.0095, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 512, #[64],
        'verbose': 1, #[1],
        'optim_method': job.optim_method,
        'Z': job.Z,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams3 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.0095, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 1024, #[64],
        'verbose': 1, #[1],
        'Z': job.Z,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'optim_method': job.optim_method,
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams4 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.04, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 4096, #[64],
        'verbose': 1, #[1],
        'optim_method': "mom",
        'Z': job.Z,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams5 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.04, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 2048, #[64],
        'verbose': 1, #[1],
        'optim_method': "mom",
        'Z': job.Z,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams6 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.04, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 512, #[64],
        'verbose': 1, #[1],
        'optim_method': "mom",
        'Z': job.Z,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams7 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.04, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 1024, #[64],
        'verbose': 1, #[1],
        'Z': job.Z,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'optim_method': "mom",
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams8 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.04, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 6144, #[64],
        'verbose': 1, #[1],
        'Z': 1e-6,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'optim_method': "mom",
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 
    hparams9 = {  # Smaller test set values
        'conv1channels': 16,
        'conv2channels': 32,
        'fc_sizes': [128,64], #layer_combos_keep,
        'losstype': job.loss, #[job.loss],
        'lr': 0.04, #[8e-5, 8e-5, 8e-5],
        'wd': 5e-5, #[5e-4],
        'batch_size': 6144, #[64],
        'verbose': 1, #[1],
        'Z': 1e-6,
        'lambda_abs': job.lambda_abs,
        'lambda_E': job.lambda_E,
        'dropout': 0.0,
        'input_channels': job.input_channels,
        'optim_method': "mom",
        'alpha_loss': job.alpha_loss,
        'n_epochs': 191, #[50]
    } 

    hparamsets = [hparams, hparams1, hparams2, hparams3, hparams4, hparams5, hparams6, hparams7,] #[hparams, hparams1, hparams2] #build_hparamsets(hparams)
    n_hparamsets = len(hparamsets)
    print('{n} hparamsets generated'.format(n=n_hparamsets))

    # Setup cluster jobs
    function = 'run_net'
    module = 'run_train_cnn'

    # Build all inputs, including adding a rng seed
    print('{} multistarts used'.format(job.n_multistarts))
    inputs = []
    job_ind = 1
    for hparamset in hparamsets:
        multistart_chunks = np.array_split(np.arange(job.n_multistarts), np.ceil(job.n_multistarts / job.n_multistarts_per_job))
        for multistart_chunk in multistart_chunks:
            rng_ind = np.random.randint(10000)
            multistart_chunk_size = len(multistart_chunk)
            hparamset_i = copy.copy(hparamset)  # remember to copy when giving each input a different set
            hparamset_i['n_multistarts'] = multistart_chunk_size
            hparamset_i['multistart_start_ind'] = rng_ind  # will be used to generate the rng_seed
            inputs.append((job_ind, data_file, job_dir, hparamset_i))
            job_ind += 1
            rng_ind += multistart_chunk_size  # advance such that each multistart gets an ind in order

    # Testing: just take the 1st few jobs
    if job.subset:
        n_subset = 5
        print('Using {} members for testing subset'.format(n_subset))
        inputs = inputs[:n_subset]

    # Save metadata
    opts_file = os.path.join(job_dir, 'opts.h5')
    h5pack.pack({'n_multistarts': job.n_multistarts,
                 'n_multistarts_per_job': job.n_multistarts_per_job,
                 'data_file': data_file,
                 'hparams': hparams,
                 'hparamsets': hparamsets,
                 'inputs': inputs,
                 'n_jobs': len(inputs)
                 }, opts_file)

    # Append current dir and parent dir to running paths to access code
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    run_paths = [curr_dir, os.path.join(curr_dir, '../')]

    # Make working dir for cluster
    working_dir = os.path.join(job_dir, 'working')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # Make dir for results
    results_dir = os.path.join(job_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Make dir for nets
    models_dir = os.path.join(job_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    opts = {
        'cluster': 'sherlock',
        'env_vars': {'OMP_NUM_THREADS': max(int(job.n_threads * 0.7), 1)},  # hack needed to set threads+over provisioning
        'env': 'source /programs/x86_64/lib/python-3.5.2-anaconda-4.1.1/bin/activate tracksml',
        'run_paths': run_paths,
        'proc': job.n_threads,
        'mem': 22,  # GB
        # 'time': 72,  # hr, a target time for all multistarts in a job
        'time': 35.5, # 1.5M train set, 50 epochs, 1 multistart/job?, time in hours
        'working_dir': working_dir,
        'name': 'train_{}'.format(job.name),
        'gpus': 4,
    }

    # Submit/run jobs
    if use_cluster:
        job_id, _ = submit_cluster(function, module, inputs, **opts)
        print('Submitted {n} jobs with job id {job_id}'.format(n=len(inputs), job_id=job_id))
    else:
        for i, input in enumerate(inputs):
            print('Running job {}/{}'.format(i+1, len(inputs)))
            run_net(*input)
            break


def run_net(ind, data_file, job_dir, opts):
    """Main function that runs in parallel

    Args:
        ind: int, run index used for bookkeeping
        data_file: str, location of hdf5 file of train/validate tracks and angles for training and basic output
        opts: dict of runtime options and hyperparameters

    Returns:
        dummy value

    Side Effects:
        Saves results (error, predictions, and convergence stats) to a separate hdf5 file with (job_ind, multistart_ind)
        Saves dumped net to a separate hdf5 file with (job_ind, multistart_ind)
    """
    print('Job {}: Training neural net on data in {} with opts {}'.format(ind, data_file, opts))

    print('Running with PyTorch version {}'.format(torch.__version__))
    
    run_dir = os.path.realpath(job_dir)

    # Load data
    #   Ignore test data
    #train = H5Dataset(data_file,'train', losstype=opts['losstype'])
    #with h5py.File(dat_file, 'r') as f:
    #    mean = f['train']['tracks_mean'][()]
    #    std = f['train']['tracks_std'][()] 
    #    maxx = f['train']['tracks_max'][()]
    mean, std = torch.load(data_file + "train/ZN.pt")
    meanE, stdE = torch.load(data_file + "train/ZNE.pt")

    train = H5Dataset(data_file + "train/", losstype=opts['losstype'],transform=transforms.Compose([ZNormalize(mean=mean,std=std) ]), energy_cal=(meanE, stdE)) 
    validate = H5Dataset(data_file + "val/", losstype=opts['losstype'],transform=transforms.Compose([ZNormalize(mean=mean,std=std) ]), energy_cal=(meanE, stdE))
    #test = H5Dataset(data_file + "test/", losstype=opts['losstype'],transform=transforms.Compose([ZNormalize(mean=mean,std=std)]), load_all=True)
    test = H5Dataset(data_file + "test/", losstype=opts['losstype'],transform=transforms.Compose([ZNormalize(mean=mean,std=std) ]), energy_cal=(meanE, stdE))
    #measured = H5DatasetEval("/scratch/groups/rwr/alpv95/data/expanded/simgen4_2p0_unpol_shift/train/", losstype=opts['losstype'],transform=transforms.Compose([ZNormalize(mean=mean,std=std), ToTensor()]), load_all=True)   
    #measured_test = H5DatasetEval(meas1_file,'train', 'meas', losstype=opts['losstype'],transform=transforms.Compose([ZNormalize(mean=train.mean,std=train.std,set_max=train.max), ToTensor()]))

    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train, batch_size=opts['batch_size'], shuffle=True, **kwargs)  #opts may actually be in opts[hparams][batch_size]?
    val_loader = torch.utils.data.DataLoader(validate, batch_size=opts['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=opts['batch_size'], shuffle=False, **kwargs)
    meas_loader = None #torch.utils.data.DataLoader(measured, batch_size=opts['batch_size'], shuffle=False, **kwargs)
    #meas_test_loader = torch.utils.data.DataLoader(measured_test, batch_size=opts['batch_size'], shuffle=False, **kwargs)

    # Set options
    opts['fc1size'] = opts['fc_sizes'][0]
    opts['fc2size'] = opts['fc_sizes'][1]
    opts['track_stats'] = True

    # Run for each multistart
    n_multistarts = opts['n_multistarts']
    multistart_start_ind = opts['multistart_start_ind']

    # Clean up opts of extraneous params
    del opts['n_multistarts']
    del opts['multistart_start_ind']
    del opts['fc_sizes']

    for i_multistart in range(n_multistarts):
        # Get this run's seed
        rng_seed = multistart_start_ind + i_multistart
        opts['rng_seed'] = rng_seed

        print('Running multistart {} with seed {}'.format(i_multistart, rng_seed))

        model_file = os.path.join(run_dir, 'models') #to save model checkpoints
        # Build and train net
        checkpoints = [mdl for mdl in os.listdir(model_file) if opts['optim_method'] + "_" + str(opts["batch_size"]) in mdl]
        if not checkpoints:
            m = TrackAngleRegressor()
        else:
            print("Loading from checkpoint! \n")
            m = TrackAngleRegressor( load_checkpoint=os.path.join(model_file, checkpoints[-1]) )
        #print(X_train.shape,y_train.shape,opts)
        train_mse, stats = m.train(train_loader, val_loader, model_file, meas_loader, **opts)  # doesn't return all train set predictions
        #train_mse, stats = m.train(train_loader, val_loader, **opts)`

        # Calculate on validation, test and measured set
        validate_mse, validate_metrics = m.test(val_loader, output_all=True)
        test_mse, test_metrics = m.test(test_loader, output_all=True)
        print('Test_mse {}'.format(test_mse))
        #meas_metrics = m.predict(meas_loader)
        #meas_test_metrics = m.predict(meas_test_loader)

        # Save results - error, predictions, and convergence stats
        results = {
            'train': {'mse': train_mse, 'stats': stats},
            'validate': {'mse': validate_mse, 'metrics': validate_metrics},
            'test': {'mse': test_mse, 'metrics': test_metrics},
           # 'meas': {'metrics': meas_metrics},
           # 'meas_test': {'metrics': meas_test_metrics}
        }
        results_file = os.path.join(run_dir, 'results', 'net{}_{}.h5'.format(ind, i_multistart))
        print(results)
        print(results_file)
        h5pack.pack(results, results_file)

        # Save the net separately to a separate location
        model_file = os.path.join(run_dir, 'models', 'net{}_{}.pt'.format(ind, i_multistart))
        m.dump(model_file)

    print('done.')
    return 0


def check_pyramidal(layer_sizes):
    """Make sure earlier layers are bigger (or equal to) than latter layers. This function is inefficient, but
    speed shouldn't matter too much based on what's passed into it."""
    n_layers = len(layer_sizes)
    if n_layers > 1:
        for i in range(1, n_layers):
            if layer_sizes[i] > layer_sizes[i - 1]:
                return False
    return True


def build_hparamsets(hparams):
    """Build hyperparameter sets for hparam optimization. Basically takes a Cartesian product to get the combos.

    Args:
        hparams: dict of {hparam name: [list of values]}

    Returns:
        list of dicts, with each dict {hparam name: value}, suitable for feeding into algorithm's options
    """
    names = list(hparams.keys())
    order = np.argsort(names)
    names = [names[i] for i in order]
    vals = [hparams[name] for name in names]
    combos = list(itertools.product(*vals))
    paramsets = []
    for combo in combos:
        opts = {}
        for i in range(len(names)):
            opts[names[i]] = combo[i]
        paramsets.append(opts)
    return paramsets


if __name__ == '__main__':
    main()
