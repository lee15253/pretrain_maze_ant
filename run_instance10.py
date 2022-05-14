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
    'agent.init_alpha=0.6', 'maze_type=AntU', 'agent.feature_dim=1024','agent.hidden_dim=1024',
    'num_pretrain_frames=1000000']
    seeds = ['seed=100', 'seed=101','seed=102', 'seed=103','seed=104', 'seed=105']
    maze_types = ['maze_type=AntU', 'maze_type=AntFb']
    num_devices = 4
    num_exp_per_device = 3
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for maze_type, seed in itertools.product(*[maze_types, seeds]):
        exp = copy.deepcopy(default)
        
        exp.append(maze_type)
        exp.append(seed)
        
        device_id = int(device % num_devices)
        exp.append(f'device_id={device_id}')
        experiments.append(exp)
        device += 1

    pool = Pool(pool_size)
    stdouts = pool.map(run_experiment, experiments, chunksize=1)
    pool.close()