import subprocess
import argparse
import json
import copy
import itertools
from multiprocessing import Pool
import time


def run_experiment(experiment):
    cmd = ['python', 'pretrain_maze.py']
    for argument in experiment:
        time.sleep(5)
        cmd.append(argument)
    
    return subprocess.check_output(cmd)


if __name__ == '__main__':
    default = ['maximum_timestep=200', 'use_wandb=true', 'agent.batch_size=512', 
    'agent.init_alpha=0.6', 'agent.feature_dim=1024','agent.hidden_dim=1024',
    'num_pretrain_frames=1010000','oracle_dur=50000', 'num_train_frames=2100000',
    'maze_type=AntU','agent.skill_dim=100', 'agent.vae_args.codebook_size=100',
    'agent.max_skill_dim=100']
    seeds = ['seed=100', 'seed=101','seed=102']
    num_devices = 4
    num_exp_per_device = 3
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for seed in seeds:
        exp = copy.deepcopy(default)
        
        exp.append(seed)
        
        device_id = int(device % num_devices)
        exp.append(f'device_id={device_id}')
        experiments.append(exp)
        device += 1

    pool = Pool(pool_size)
    stdouts = pool.map(run_experiment, experiments, chunksize=1)
    pool.close()