import os
import simreal
import sys

from simreal import EpisodeRunner
from simreal import RandomAgent
from simreal.utils import pickle_dump
from simreal.utils import set_global_seeds
from simreal.utils import numpify
from simreal.rlbase import Config
from simreal.rlbase import Grid
from simreal.rlbase import run_experiment
from simreal.wrappers import RecordEpisodeStatistics
from simreal.wrappers import TimeStepEnv

from simreal.baselines.SAC import NormalizeAction
from simreal.baselines.SAC import Agent
from simreal.baselines.SAC import Engine
from simreal.baselines.SAC import ReplayBuffer

import torch
import random

config = Config(
    {'log.freq': 10,
     'checkpoint.num': 10,
     
     'env.id': Grid(['CustomPendulum-v0']),
     
     'agent.gamma': 0.99,
     'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
     'agent.actor.lr': 3e-4, 
     'agent.actor.use_lr_scheduler': False,
     'agent.critic.lr': 3e-4,
     'agent.critic.use_lr_scheduler': False,
     'agent.initial_temperature': 1.0,
     'agent.max_grad_norm': 999999,  # grad clipping by norm
     
     'replay.capacity': 1000000, 
     'replay.init_trial': 10,  # number of random rollouts initially
     'replay.batch_size': 256,
     
     'train.timestep': 100000,  # total number of training (environmental) timesteps
     'eval.num': 100
    })


def make_env(config, seed, mode):
    assert mode in ['train', 'eval']
    env = simreal.make(config['env.id'])
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    env = NormalizeAction(env)
    if mode == 'eval':
        env = RecordEpisodeStatistics(env, deque_size=100)
    env = TimeStepEnv(env)
    return env


def run(config, seed, device, logdir):
    set_global_seeds(seed)
    
    env = make_env(config, seed, 'train')
    eval_env = make_env(config, seed, 'eval')
    random_agent = RandomAgent(config, env, device)
    agent = Agent(config, env, device)
    runner = EpisodeRunner()
    replay = ReplayBuffer(env, config['replay.capacity'], device)
    engine = Engine(config, agent=agent, random_agent=random_agent, env=env, eval_env=eval_env, runner=runner, replay=replay, logdir=logdir)
    
    train_logs, eval_logs = engine.train()
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
    return None  

def test(goal_steps, n_tests, actor, render = False):
    scores = []
    actions = []
    for each_game in range(n_tests):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            if render:
                env.render()
            if len(prev_obs) == 0:
                action = [random.uniform(-2,2)]   #first action is a random torque between the max torque boundaries
            else:
                obs = torch.as_tensor(prev_obs, dtype=torch.float32).to(device)
                action = numpify(actor(obs).sample(), 'float')


            actions.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        scores.append(score)
    print('Average Score: {}'.format(sum(scores)/len(scores)))

if __name__ == '__main__':

    TRAIN = False
    RUN_ON_GPU = True
    TEST_MODEL = True

    device = torch.device("cpu")
    if RUN_ON_GPU:
        device = torch.device("cuda:0")

    if TRAIN:
        run_experiment(run=run,
                       config=config,
                       seeds=[3949341511],
                       log_dir='logs_experiment/default',
                       max_workers=os.cpu_count(),
                       chunksize=1,
                       use_gpu=RUN_ON_GPU,  # GPU much faster, note that performance differs between CPU/GPU
                       gpu_ids=None)

    if TEST_MODEL:
        env = simreal.make('CustomPendulum-v0')
        agent = torch.load('logs_experiment/default/0/3949341511/model_agent_334.pt')
        test(goal_steps=500, n_tests=10, actor=agent, render=True)
        env.close()