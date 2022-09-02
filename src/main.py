import os, sys
import json
from simreal.rlbase import Config, Grid, run_experiment
from simreal.sim2real.adaptive_domain_randomization import ADR


# PPO Config
config_PPO = {
     'log.freq': 5,
     'checkpoint.num': 10,

     'env.id': ['CustomAcrobot-v1'],
     'env.normalize_obs': True,
     'env.normalize_reward': True,

     'nn.sizes': [64, 64],

     'agent.policy_lr': 3e-4,
     'agent.use_lr_scheduler': False,
     'agent.value_lr': 1e-3,
     'agent.gamma': 0.99,
     'agent.gae_lambda': 0.95,
     'agent.standardize_adv': True,  # standardize advantage estimates
     'agent.max_grad_norm': 0.5,  # grad clipping by norm
     'agent.clip_range': 0.2,  # ratio clipping

     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     'agent.std0': 0.6,  # initial std

     'train.timestep': 100000, # 150000, # total number of training (environmental) timesteps
     'train.timestep_per_iter': 2048,  # number of timesteps per iteration
     'train.batch_size': 64,
     'train.num_epochs': 10,

     'eval.num': 50
     }


# initial env args
args_0 = {
     'domain_randomization_type': 'random',
     'randomize_every': 'episode',
     'frequency': 1,
     'params': {'l1': ('mean_std', [0.22, 0.05]),
                'l2': ('mean_std', [0.27, 0.05]),
                'm1': ('mean_std', [3.5, .50]),
                'm2': ('mean_std', [0.9, 0.20])},
     'distribution': 'normal'}


# CEM Config
config_CEM = {
     'log.freq': 1,

     'env.id': ['CustomAcrobot-v1'],

     'env.clip_action': True,  # clip action within valid bound before step()

     'train.generations': 3,  # total number of es generations
     'train.popsize': 1000,
     'train.worker_chunksize': 100,  # must be divisible by popsize

     # define initial phi
     'train.mu0': [0.22, 0.27, 3.5, 0.9],
     'train.std0': [0.05, .05, .50, .20],

     'train.elite_amount': 100,  # should be significantly smaller then popsize
     }


def check_logdir(logdir):
     if not os.path.exists(logdir):
        os.makedirs(logdir)
     if len(os.listdir(logdir)) != 0:
        proceed = input('Experiment directory not empty, continue? [y/n]: ')
        if proceed != 'y':
            print('Aborted')
            sys.exit()


def copy_configs(logdir):
     check_logdir(logdir)

     with open(f'{logdir}/config_PPO.json', 'w') as outfile:
          json.dump(config_PPO, outfile)
     with open(f'{logdir}/args_0.json', 'w') as outfile:
          json.dump(args_0, outfile)
     with open(f'{logdir}/config_CEM.json', 'w') as outfile:
          json.dump(config_CEM, outfile)


if __name__ == '__main__':

     experiment_name = "acrobot-baseline"
     seeds = [1084389005, 1549325168, 1781137871, 2010844016, 2128474801]            # List of seeds

     logdir = 'src/logs/{}'.format(experiment_name)
     copy_configs(logdir)

     adr = ADR(iter_max=10,
               config_PPO=Config(config_PPO),
               args_0=args_0,
               config_CEM=Config(config_CEM),
               logdir=logdir,
               run_on_gpu=False,
               seed=seeds)

     adr.run()