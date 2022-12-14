import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle

import gym.spaces as spaces

from simreal import BaseAgent
from simreal.utils import pickle_dump
from simreal.utils import tensorify
from simreal.utils import numpify
from simreal.networks import Module
from simreal.networks import make_fc
from simreal.networks import ortho_init
from simreal.networks import CategoricalHead
from simreal.networks import DiagGaussianHead
from simreal.networks import linear_lr_scheduler
from simreal.utils import bootstrapped_returns
from simreal.utils import gae
from simreal.utils import explained_variance as ev
from simreal.utils import describe

from torch.utils.data import DataLoader
from simreal.baselines.PPO.dataset import Dataset


class Actor(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.feature_layers = make_fc(spaces.flatdim(env.observation_space), config['nn.sizes'])
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='tanh', constant_bias=0.0)
        
        feature_dim = config['nn.sizes'][-1]
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = CategoricalHead(feature_dim, env.action_space.n, device, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            self.action_head = DiagGaussianHead(feature_dim, spaces.flatdim(env.action_space), device, config['agent.std0'], **kwargs)
        
        self.to(self.device)
        
    def forward(self, x):
        for layer in self.feature_layers:
            x = torch.tanh(layer(x))
        action_dist = self.action_head(x)
        return action_dist


class Critic(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.feature_layers = make_fc(spaces.flatdim(env.observation_space), config['nn.sizes'])
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        
        feature_dim = config['nn.sizes'][-1]
        self.V_head = nn.Linear(feature_dim, 1)
        ortho_init(self.V_head, weight_scale=1.0, constant_bias=0.0)
        
        self.to(self.device)
        
    def forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        V = self.V_head(x)
        return V


class Agent(BaseAgent):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(config, env, device, **kwargs)

        self.start_time = time.perf_counter()
        
        self.policy = Actor(config, env, device, **kwargs)
        self.value = Critic(config, env, device, **kwargs)
        
        self.total_timestep = 0
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config['agent.policy_lr'])
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config['agent.value_lr'])
        if config['agent.use_lr_scheduler']:
            self.policy_lr_scheduler = linear_lr_scheduler(self.policy_optimizer, config['train.timestep'], min_lr=1e-8)
        
    def choose_action(self, x, **kwargs):
        obs = tensorify(x.observation, self.device).unsqueeze(0)
        action_dist = self.policy(obs)
        V = self.value(obs)
        action = action_dist.sample()
        out = {}
        out['action_dist'] = action_dist
        out['V'] = V
        out['entropy'] = action_dist.entropy()
        out['action'] = action
        out['raw_action'] = numpify(action, self.env.action_space.dtype).squeeze(0)
        out['action_logprob'] = action_dist.log_prob(action.detach())
        return out
    
    def learn_one_update(self, data):
        data = [d.detach().to(self.device) for d in data]
        observations, old_actions, old_logprobs, old_entropies, old_Vs, old_Qs, old_As = data
        
        action_dist = self.policy(observations)
        logprobs = action_dist.log_prob(old_actions).squeeze()
        entropies = action_dist.entropy().squeeze()
        Vs = self.value(observations).squeeze()
        assert all([x.ndim == 1 for x in [logprobs, entropies, Vs]])
        
        ratio = torch.exp(logprobs - old_logprobs)
        eps = self.config['agent.clip_range']
        policy_loss = -torch.min(ratio*old_As, 
                                 torch.clamp(ratio, 1.0 - eps, 1.0 + eps)*old_As)
        policy_loss = policy_loss.mean(0)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['agent.max_grad_norm'])
        self.policy_optimizer.step()
        if self.config['agent.use_lr_scheduler']:
            self.policy_lr_scheduler.step(self.total_timestep)
        
        clipped_Vs = old_Vs + torch.clamp(Vs - old_Vs, -eps, eps)
        value_loss = torch.max(F.mse_loss(Vs, old_Qs, reduction='none'), 
                               F.mse_loss(clipped_Vs, old_Qs, reduction='none'))
        value_loss = value_loss.mean(0)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        value_grad_norm = nn.utils.clip_grad_norm_(self.value.parameters(), self.config['agent.max_grad_norm'])
        self.value_optimizer.step()
        
        out = {}
        out['policy_grad_norm'] = policy_grad_norm
        out['value_grad_norm'] = value_grad_norm
        out['policy_loss'] = policy_loss.item()
        out['policy_entropy'] = entropies.mean().item()
        out['value_loss'] = value_loss.item()
        out['explained_variance'] = ev(y_true=numpify(old_Qs, 'float'), y_pred=numpify(Vs, 'float'))
        out['approx_kl'] = (old_logprobs - logprobs).mean(0).item()
        out['clip_frac'] = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float().mean(0).item()
        return out
        
    def learn(self, D, **kwargs):
        logprobs = [torch.cat(traj.get_infos('action_logprob')) for traj in D]
        entropies = [torch.cat(traj.get_infos('entropy')) for traj in D]
        Vs = [torch.cat(traj.get_infos('V')) for traj in D]
        with torch.no_grad():
            last_observations = tensorify([traj[-1].observation for traj in D], self.device)
            last_Vs = self.value(last_observations).squeeze(-1)
        Qs = [bootstrapped_returns(self.config['agent.gamma'], traj.rewards, last_V, traj.reach_terminal)
              for traj, last_V in zip(D, last_Vs)]
        As = [gae(self.config['agent.gamma'], self.config['agent.gae_lambda'], traj.rewards, V, last_V, traj.reach_terminal)
              for traj, V, last_V in zip(D, Vs, last_Vs)]
        
        # Metrics -> Tensor, device
        logprobs, entropies, Vs = map(lambda x: torch.cat(x).squeeze(), [logprobs, entropies, Vs])
        Qs, As = map(lambda x: tensorify(np.concatenate(x).copy(), self.device), [Qs, As])
        if self.config['agent.standardize_adv']:
            As = (As - As.mean())/(As.std() + 1e-4)
        assert all([x.ndim == 1 for x in [logprobs, entropies, Vs, Qs, As]])
        
        dataset = Dataset(D, logprobs, entropies, Vs, Qs, As)
        dataloader = DataLoader(dataset, self.config['train.batch_size'], shuffle=True)
        for epoch in range(self.config['train.num_epochs']):
            logs = [self.learn_one_update(data) for data in dataloader]

        self.total_timestep += sum([traj.T for traj in D])
        out = {}
        if self.config['agent.use_lr_scheduler']:
            out['current_lr'] = self.policy_lr_scheduler.get_lr()

        out['policy_grad_norm'] = np.mean([float(item['policy_grad_norm']) for item in logs])
        out['value_grad_norm'] = np.mean([float(item['value_grad_norm']) for item in logs])
        out['policy_loss'] = np.mean([item['policy_loss'] for item in logs])
        out['policy_entropy'] = np.mean([item['policy_entropy'] for item in logs])
        out['value_loss'] = np.mean([item['value_loss'] for item in logs])
        out['explained_variance'] = np.mean([item['explained_variance'] for item in logs])
        out['approx_kl'] = np.mean([item['approx_kl'] for item in logs])
        out['clip_frac'] = np.mean([item['clip_frac'] for item in logs])
        return out
    
    def checkpoint(self, logdir, num_iter):
        torch.save(self.policy, logdir/f'model_agent_{num_iter}.pt')
        pickle.dump(self, open(logdir/f'PPO_agent_{num_iter}.pkl', 'wb'))
        if self.config['env.normalize_obs']:
            moments = (self.env.obs_moments.mean, self.env.obs_moments.var)
            pickle_dump(obj=moments, f=logdir/f'obs_moments_{num_iter}', ext='.pth')
