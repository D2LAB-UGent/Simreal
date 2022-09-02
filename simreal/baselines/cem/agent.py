import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym.spaces as spaces
from simreal import BaseAgent
from simreal.utils import pickle_dump
from simreal.utils import tensorify
from simreal.utils import numpify
from simreal.networks import Module
from simreal.networks import make_fc
from simreal.networks import CategoricalHead
from simreal.networks import DiagGaussianHead


class MLP(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.feature_layers = make_fc(spaces.flatdim(env.observation_space), config['nn.sizes'])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['nn.sizes']])
        self.to(self.device)
        
    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.relu(layer(x)))
        return x


class Agent(BaseAgent):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(config, env, device, **kwargs)
        
        self.feature_network = MLP(config, env, device, **kwargs)
        feature_dim = config['nn.sizes'][-1]
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = CategoricalHead(feature_dim, env.action_space.n, device, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            self.action_head = DiagGaussianHead(feature_dim, spaces.flatdim(env.action_space), device, config['agent.std0'], **kwargs)
        self.total_timestep = 0
        
    def choose_action(self, x, **kwargs):
        obs = tensorify(x.observation, self.device).unsqueeze(0)
        features = self.feature_network(obs)
        action_dist = self.action_head(features)
        action = action_dist.sample()
        out = {}
        out['raw_action'] = numpify(action, self.env.action_space.dtype).squeeze(0)
        return out
    
    def learn(self, D, **kwargs):
        pass
    
    def checkpoint(self, logdir, num_iter):
        self.save(logdir/f'agent_{num_iter}.pth')
        if 'env.normalize_obs' in self.config and self.config['env.normalize_obs']:
            moments = (self.env.obs_moments.mean, self.env.obs_moments.var)
            pickle_dump(obj=moments, f=logdir/f'obs_moments_{num_iter}', ext='.pth')
