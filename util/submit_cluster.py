# Simple Python API for submitting job arrays to the cluster
# Works similarly to the Matlab submitCluster.m
# Notes:
#   This only works for Python 3 right now
#   Function inputs and outputs are packed using msgpack (and msgpack-numpy) right now. This imposes some limitations on
#       what can be packed, but is safer/cleaner/more compatible between Python vers.
#       An alternative is to use pickle for this.
# See Torque options at: http://docs.adaptivecomputing.com/torque/4-1-3/Content/topics/2-jobs/requestingRes.htm
#
# Note: To test c3ddb, it may be useful to get an interactive session on a node:
# srun --pty --mem 16000 -p sched_mem1TB -t 1:00:00 /bin/bash
# srun --pty --mem 16000 --gres=gpu:1 -p sched_mem1TB -t 1:00:00 /bin/bash

#Now edited by ALP, Kevin version runs from local for MIT clusters, ALP version runs from cluster on cluster - Sherlock $GROUP_HOME


import os
import getpass
import datetime
import numbers
from math import floor
import subprocess
import numpy as np  # For test job
import pickle
import time as ptime

# Cluster resources and defaults
# TODO: list of all available queues, max procs, etc. for error checking
resources = {
#    'thor': {
#        'queue_manager': 'torque',
#        'default_queue': 'default',
#        'default_env': 'source /programs/x86_64/lib/python-3.5.2-anaconda-4.1.1/bin/activate default',
#        'is_remote': False,
#    },
#    'vali': {
#        'queue_manager': 'torque',
#        'default_queue': 'default',
#        'default_env': 'source /programs/x86_64/lib/python-3.5.2-anaconda-4.1.1/bin/activate default',
#        'is_remote': False,
#    },
#    'c3ddb': {
#        'queue_manager': 'slurm',
#        'default_queue': 'defq',
#        'default_env': 'source /scratch/users/kshi/programs/anaconda3/bin/activate default-env',
#        'is_remote': True,
#    }
    'sherlock': {
        'queue_manager': 'slurm',
        'default_queue': 'owners',
#        'default_env': 'source /scratch/users/kshi/programs/anaconda3/bin/activate default-env',
        'is_remote': False,
    }
}

queue_resources = {
    'torque': {
        'submit_cmd': 'qsub',
        'cancel_cmd': 'qdel',
    },
    'slurm': {
        'submit_cmd': 'sbatch',
        'cancel_cmd': 'scancel',
    }
}


def fix_time(hours, queue_manager):
    """Convert time in hours to standard d:hh:mm:ss or d-hh:mm:ss format expected by queue manager"""
    if not isinstance(hours, numbers.Number) or hours <= 0:
        raise ValueError('Invalid time')
    hours = float(hours)
    d = floor(hours/24)
    _, h_ = divmod(hours - 24.0*d, 24.0)
    h = floor(h_)
    _, m_ = divmod(hours*60.0 - (60.0*24.0*d+60.0*h), 60.0)
    m = floor(m_)
    _, s_ = divmod(hours*60.0*60.0 - (60.0*60.0*24.0*d+60.0*60.0*h+60.0*m), 60.0)
    s = floor(s_)

    if queue_manager == 'torque':
        sep = ':'
    elif queue_manager == 'slurm':
        sep = '-'
    else:
        raise ValueError('Queue manager not recognized')

    return '%d%s%2.2d:%2.2d:%2.2d' %(d, sep, h, m, s)


def fix_gpu(gpus, queue_manager):
    """Convert GPU spec to string, if needed."""
    if queue_manager != 'slurm' or gpus == 0:
        return ''
    return '#SBATCH --gres=gpu:{n}\n'.format(n=gpus)


def submit(function, module, inputs,
           cluster='sherlock',
           queue=None,  # will be assigned cluster's default queue if empty
           env_vars={},
           env=None,  # will be assigned cluster's default env script if empty
           run_paths=[],
           proc=1,
           full_node=False,
           props=None,
           mem=3.75,
           time=8,
           max_concurrent=0,
           gpus=4,
           working_dir='$GROUP_HOME/alpv95/tracksml',
           remote_dir=None,
           user='alpv95',
           name=None):
    """

    Args:

        function (string): Name of function to run in parallel on cluster
        module (string): Name of module that contains above function
        inputs (iterable of iterables): function arguments, where the outer list holds each job and the inner lists are the
            args for each job. Note: Only accepts positional args right now
        cluster (string): Cluster to submit to. Currently supports thor and vali (local clusters) and c3ddb (remote clusters)
        queue (string): Name of queue on cluster. Local clusters only have the default queue 'default'. c3ddb calls
            these partitions and uses 'defq' as default.
        env_vars (dict): Environment vars to set in bash. These are exported.
        env (string): Command to activate Python environment. This will usually be some sort of "source" command. All
            clusters use an Anaconda Python 3.5, but in different locations.
        run_paths (list of strings): Additional directories to add to the path. Needed to load modules in other dirs.
            For remote clusters, the files (but not subdirs) in each run_path are copied over to a dummy dir in
            remote_dir/sources and those paths added. This is a hack to make paths/sources available to the minimal
            running env.
        proc (positive int): Number of CPU cores per job
        full_node (bool): Whether to get the entire node exclusively. You can set any number of proc. Testing out for
            Torque only for now.
        props (string or None): Node properties required. Default is None, but 'avx2' is often useful. Specify multiple
            properties with colon separators: '<prop1>:<prop2>:...'.
        mem (positive float): Amount of RAM per job in GiB
        time (positive float): Max walltime per job in hours
        max_concurrent (positive int): Maximum concurrent jobs in job array to run at a time. May be useful in being
            nice but rarely useful.
        gpus (nonnegative int): Number of GPUs (for c3ddb, sched_mem1TB partition only right now). Default is 0.
        working_dir (string): Location of inputs, outputs, logs, and other files for controlling jobs. Default is an
            auto-generated dir based on the submission time and the job name.
        remote_dir (string): For a remote cluster, a directory to hold inputs, outputs, logs, and other files for
            controlling jobs. Files in the local working_dir will be copied over to remote_dir before starting the job,
            and a copy-back script will be added to working_dir to facilitate copying results back. Default for c3ddb
            is an auto-generated dir like working_dir, but in /scratch/users/<username>.
        user (string): Username to password-less SSH into the cluster head node and submit jobs under. Default is your
            current username.
        name (string) Name of job

    Returns:
        job_id (string): Queue manager job id
        working_dir (string): Location of all files, including results
    """
    n_jobs = len(inputs)

    if name is None:
        name = function

    #if user is None:
        #user = getpass.getuser()

   # if working_dir is None:
   #     dir_str = datetime.datetime.now().strftime('%Y.%m.%d-%H.%M') + '_' + name
   #     # dir_str = '000_test_' + job_name  # consistent dir for testing
   #     working_dir = os.path.join('/data', user, dir_str)

    # Prep working directory and function inputs

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    local_dir = working_dir

    for i in range(n_jobs):
        input_file = os.path.join(working_dir,'input{i}'.format(i=i+1))
        with open(input_file, 'wb') as f:
            pickle.dump(inputs[i], f)

    # Torque vs Slurm options
    if cluster in resources:
        queue_manager = resources[cluster]['queue_manager']
        submit_command = queue_resources[queue_manager]['submit_cmd']
        cancel_command = queue_resources[queue_manager]['cancel_cmd']
        is_remote = resources[cluster]['is_remote']
    else:
        raise ValueError('Cluster not recognized')

    if queue is None:
        queue = resources[cluster]['default_queue']

    #if env is None:
        #env = resources[cluster]['default_env']

    # Handle node properties, if required
    if props is None:
        props_str = ''
    else:
        props_str = ':' + props

    # Handle max concurrent job spec, if desired
    if max_concurrent == 0:
        max_concurrent_str = ''
    else:
        max_concurrent_str = '%{n}'.format(n=max_concurrent)

    # Handle use of full node, if needed

    if full_node:
        full_node_str = '\n#PBS -l naccesspolicy=SINGLEJOB -n'
    else:
        full_node_str = ''

    # Modify run_paths for remote clusters
   # if is_remote:
   #     if remote_dir is None:
   #         remote_dir = os.path.join('/scratch/users', user, dir_str)
   #     remote_run_paths = []
   #     for entry in run_paths:
   #         # Strip absolute path in order to join, if needed
   #         if entry[0] == '/':
   #             entry = entry[1:]
   #         remote_run_paths.append(os.path.join(remote_dir, 'sources', entry))
   #     local_run_paths = run_paths
   #     run_paths = remote_run_paths
   #     working_dir = remote_dir

    # Make env vars setting string
   # env_vars_str = ''
   # for var_name, var_val in env_vars.items():
   #     env_vars_str += 'export {}={}\n'.format(var_name, var_val)

    # Construct job options dictionary for submit files
    job_opts = {
        'user': user,
        'cluster': cluster,
        'queue': queue,
        'proc': proc,
        'full_node_str': full_node_str,
        'props': props_str,
        'mem': int(mem * 1024),  # in MiB
        'time': fix_time(time, queue_manager),  # queue manager's time string
        'name': name,
        'n_jobs': n_jobs,
        'max_concurrent_str': max_concurrent_str,
        'gpu_str': fix_gpu(gpus, queue_manager),
        #'env_vars_str': env_vars_str,
        #'env_setup': env,
        'working_dir': working_dir,
        'local_dir': local_dir,
        #'remote_dir': remote_dir,
        'run_path': run_paths,  # this must be a list of paths
        'run_module': module,
        'run_function': function,
        'submit_command': submit_command
    }
    print(job_opts['time'])
    # Make run files
    job_file = os.path.join(local_dir, 'job.sh')
    run_file = os.path.join(local_dir, 'run_file.py')
    id_file = os.path.join(local_dir, 'jobname.txt')
    kill_file = os.path.join(local_dir, 'killall.sh')

    # Make run file for each job array component
    #   Add path of scripts directly and then do imports as normal
    with open(run_file, 'w') as f:
        f.write("""# Main running script for job array components
import sys
sys.path.extend({run_path})
import pickle
from {run_module} import {run_function}

i = sys.argv[1]
input_file = 'input{{i}}'.format(i=i)
output_file = 'output{{i}}'.format(i=i)

with open(input_file, 'rb') as f:
    inputs = pickle.load(f)

outputs = {run_function}(*inputs)

with open(output_file, 'wb') as f:
    pickle.dump(outputs, f)
""".format(**job_opts))

    # Make main job file
    if queue_manager == 'torque':
        with open(job_file, 'w') as f:
            f.write("""#!/usr/bin/env bash
#PBS -j oe
#PBS -q {queue}
#PBS -r n
#PBS -l nodes=1:ppn={proc}{props}{full_node_str}
#PBS -l mem={mem}Mb
#PBS -l walltime={time}
#PBS -N {name}
#PBS -t 1-{n_jobs}{max_concurrent_str}

{env_vars_str}
{env_setup}
cd {working_dir}
python run_file.py $PBS_ARRAYID
""".format(**job_opts))

    if queue_manager == 'slurm':
        with open(job_file, 'w') as f:
            f.write("""#!/bin/bash
#SBATCH --output=out_%j.out
#SBATCH --partition={queue}
#SBATCH --nodes=1
#SBATCH --ntasks={proc}
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu={mem}M
#SBATCH --time={time}
#SBATCH --job-name={name}
#SBATCH --array=1-{n_jobs}{max_concurrent_str}
{gpu_str}

module load python/3.6.1
module load py-numpy/1.14.3_py36
##module load py-pytorch/1.0.0_py36
module load viz
module load py-matplotlib/2.1.2_py36

cd {working_dir}
python3 run_file.py $SLURM_ARRAY_TASK_ID
""".format(**job_opts))
#SBATCH --partition={queue}

    # Copy inputs, scripts, and run_path contents for remote clusters
   # if is_remote:
   #     # Copy inputs
   #     print('Copying working dir to remote cluster...')
   #     out = subprocess.run('scp -r {local_dir} {cluster}:{remote_dir}'.format(**job_opts),
   #                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
   #     if out.returncode != 0:
   #         raise RuntimeError(
   #             'Copying working dir to remote cluster failed. Message: {msg}'.format(msg=out.stderr.decode('utf-8')))
   #     print('done.')

   #     # Copy scripts
   #     print('Copying sources to remote cluster...')
   #     for i in range(len(local_run_paths)):
   #         local = local_run_paths[i]
   #         remote = remote_run_paths[i]  # equal to run_paths now
   #         out = subprocess.run('ssh {user}@{cluster} "mkdir -p {remote}" && scp {local}/* {cluster}:{remote}'.format(
   #             local=local, remote=remote, user=job_opts['user'], cluster=job_opts['cluster']),
   #             stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # non-recursive, only copy files
   #         # Ignore errors - these "not a regular file" are directories that should be skipped
   #         # if out.returncode != 0:
   #         #     raise RuntimeError(
   #         #         'Copying source directory {local}/* failed. Message: {msg}'.format(local=local, msg=out.stderr.decode('utf-8')))
   #     print('done.')

    # Submit jobs
    out = subprocess.run('cd {working_dir} && {submit_command} job.sh'.format(**job_opts),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Check status
    if out.returncode != 0:
        raise RuntimeError('Submission to the queue manager failed. Message: {msg}'.format(msg=out.stderr.decode('utf-8')))

    # Get Job ID
    job_id = out.stdout.decode('utf-8').strip()

    # Util script for viewing job ID
    with open(id_file, 'w') as f:
        f.write("""{id}""".format(id=job_id))

    # Util script for killing all jobs
#    job_opts['cancel_command'] = cancel_command
#    job_opts['job_id'] = job_id
#    with open(kill_file, 'w') as f:
#        f.write("""#!/usr/bin/env bash
#ssh {user}@{cluster} {cancel_command} {job_id}
#""".format(**job_opts))
#    out = subprocess.run('chmod +x {kill_file}'.format(kill_file=kill_file),
#                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Util script for copying back results from remote cluster
#    if is_remote:
#        copy_back_file = os.path.join(local_dir, 'copy_results.sh')
#        with open(copy_back_file, 'w') as f:
#            f.write("""#!/usr/bin/env bash
#rsync -avz --exclude 'sources' -e ssh {user}@{cluster}:{remote_dir}/* {local_dir}/
#""".format(**job_opts))
#        out = subprocess.run('chmod +x {copy_back_file}'.format(copy_back_file=copy_back_file),
#                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    return job_id, local_dir


def test_fun(a, b):
    """Example function to run on the cluster"""
    print('a = {a}, b = {b}'.format(a=a, b=b))
    x = np.random.rand(a, a)
    start = ptime.time()
    for i in range(b):
        x = np.dot(x, x)
    end = ptime.time()
    print(str(end - start) + 's taken to run test matmuls')
    return x


def main():
    function = 'test_fun'
    module = 'submit_cluster'

    inputs = []
    for i in range(3):
        inputs.append([i, 2 * i])

    # Append current dir to run_paths since test_fun is in this file
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # All default options except shorten max time
    opts = {
        # 'cluster': 'c3ddb',
        # 'queue': 'sched_mem1TB',
        'cluster': 'thor',
        'run_paths': [curr_dir],
        'time': 1  # hr
    }

    # Submit jobs
    status = submit(function, module, inputs, **opts)
    print(status)


# Test code
if __name__ == '__main__':
    main()
