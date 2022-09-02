from multiprocessing import Pool
import time
import os

import numpy as np
import torch
import gym

import simreal
from simreal import Logger
from simreal import EpisodeRunner
from simreal.utils import describe
from simreal.utils import CloudpickleWrapper
from simreal.utils import pickle_dump
from simreal.utils import tensorify
from simreal.utils import set_global_seeds
from simreal.rlbase import Config
from simreal.rlbase import Grid
from simreal.rlbase import run_experiment
from simreal.wrappers import TimeStepEnv

from simreal import CEM
from simreal.baselines.cem.agent import Agent


config = Config(
    {'log.freq': 10, 
     'checkpoint.num': 3,
     
     'env.id': Grid(['CustomAcrobot-v1']),
     
     'nn.sizes': [64, 64],
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     'agent.std0': 0.6,  # initial std
     
     'train.generations': 500,  # total number of ES generations
     'train.popsize': 32,
     'train.worker_chunksize': 4,  # must be divisible by popsize
     'train.mu0': 0.0,
     'train.std0': 1.0,
     'train.elite_ratio': 0.2,
     'train.noise_scheduler_args': [0.01, 0.001, 400, 0]  # [initial, final, N, start]
    })


def make_env(config, seed, mode):
    assert mode in ['train', 'eval']
    env = simreal.make(config['env.id'])
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    if config['env.clip_action'] and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    env = TimeStepEnv(env)
    return env


def fitness(data):
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    config, seed, device, param = data
    env = make_env(config, seed, 'train')
    agent = Agent(config, env, device)
    agent.from_vec(tensorify(param, 'cpu'))
    runner = EpisodeRunner()
    with torch.no_grad():
        D = runner(agent, env, 10)
    R = np.mean([sum(traj.rewards) for traj in D])
    H = np.mean([traj.T for traj in D])

    print(R)
    return R, H


def run(config, seed, device, logdir, args):
    set_global_seeds(seed)
    torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
    
    print('Initializing...')
    agent = Agent(config, make_env(config, seed, 'eval'), device)
    es = CEM([config['train.mu0']]*agent.num_params, config['train.std0'], 
             {'popsize': config['train.popsize'], 
              'seed': seed, 
              'elite_ratio': config['train.elite_ratio'], 
              'noise_scheduler_args': config['train.noise_scheduler_args']})
    train_logs = []
    checkpoint_count = 0
    with Pool(processes=config['train.popsize']//config['train.worker_chunksize']) as pool:
        print('Finish initialization. Training starts...')
        for generation in range(config['train.generations']):
            t0 = time.perf_counter()
            solutions = es.ask()

            print(solutions.shape)

            data = [(config, seed, device, solution) for solution in solutions]
            out = pool.map(CloudpickleWrapper(fitness), data, chunksize=config['train.worker_chunksize'])
            Rs, Hs = zip(*out)

            print(Rs)
            print(len(Rs))

            es.tell(solutions, [-R for R in Rs])
            logger = Logger()
            logger('generation', generation+1)
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            logger('Returns', describe(Rs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('Horizons', describe(Hs, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('fbest', es.result.fbest)
            train_logs.append(logger.logs)
            if generation == 0 or (generation+1) % config['log.freq'] == 0:
                logger.dump(keys=None, index=0, indent=0, border='-'*50)
            if (generation+1) >= int(config['train.generations']*(checkpoint_count/(config['checkpoint.num'] - 1))):
                agent.from_vec(tensorify(es.result.xbest, 'cpu'))
                agent.checkpoint(logdir, generation+1)
                checkpoint_count += 1
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    return None

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config,
                   args=None,
                   seeds=[1770966829],
                   log_dir='logs/default',
                   max_workers=os.cpu_count(),  # tune to fulfill computation power
                   chunksize=1, 
                   use_gpu=False,
                   gpu_ids=None)
