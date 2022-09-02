import torch
import pickle

import simreal
from simreal import DomainRandomization
from utils import test
 
args = {
     'domain_randomization_type': 'random',
     'randomize_every': 'variation',
     'frequency': 20,
     'params': {'l1': ('mean_std', [0.22, 0.05]),
                'l2': ('mean_std', [0.27, 0.05]),
                'm1': ('mean_std', [3.5, .50]),
                'm2': ('mean_std', [0.9, 0.20])},
     'distribution': 'normal'}
 

if __name__ == '__main__':

     env = simreal.make('CustomAcrobot-v1')
     env = DomainRandomization(env, args)

     # Load an available policy    
     dir = f'src/logs/acrobot-baseline/PPO-0/0/1084389005'
     random_agent = torch.load(f'{dir}/model_agent_1.pt')
     try:
          moments = pickle.load(open(f'{dir}/obs_moments_1.pth', 'rb'))
     except:
          moments = None


     test(n_tests=1, policy=random_agent, env=env, moments=moments, render=True)