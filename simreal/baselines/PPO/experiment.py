import os, sys
from itertools import count
import simreal
import time
import torch
import gym

from simreal import DomainRandomization
from simreal import StepRunner, EpisodeRunner
from simreal.utils import pickle_dump, set_global_seeds, numpify
from simreal.rlbase import Config, Grid, run_experiment
from simreal.wrappers import RecordEpisodeStatistics, NormalizeObservation, NormalizeReward
from simreal.wrappers import TimeStepEnv, Noisy

from simreal.baselines.PPO.agent import Agent
from simreal.baselines.PPO.engine import Engine

args = {
    'domain_randomization_type': 'random',
    'randomize_every': 'episode',
    'frequency': 1,
    'params': { 'm': ('min_max', [.9, 1.1, 1.0]),
                'l': ('min_max', [.45, .55, .50])},
    'distribution': 'uniform'}

config = Config(
    {'log.freq': 1,
     'checkpoint.num': 3,
     
     'env.id': Grid(['CustomPendulum-v0']),
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
     
     'train.timestep': 100000,  # total number of training (environmental) timesteps
     'train.timestep_per_iter': 2048,  # number of timesteps per iteration
     'train.batch_size': 64,
     'train.num_epochs': 10
    })

def make_env(config, seed, mode):
    assert mode in ['train', 'eval']
    env = simreal.make(config['env.id'])
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    if config['env.clip_action'] and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    if mode == 'train':
        env = DomainRandomization(env, args)
        env = RecordEpisodeStatistics(env, deque_size=100)
        if config['env.normalize_obs']:
            env = NormalizeObservation(env, clip=5.)
        if config['env.normalize_reward']:
            env = NormalizeReward(env, clip=10., gamma=config['agent.gamma'])
    if mode == 'eval':
        env = Noisy(env, max_actuation_noise_perc=0, max_observation_noise_perc=0, max_offset_perc=0, friction=0)
        env = RecordEpisodeStatistics(env, deque_size=100)
    env = TimeStepEnv(env)
    return env

def run(config, seed, device, logdir, args):
    set_global_seeds(seed)
    
    env = make_env(config, seed, 'train')
    agent = Agent(config, env, device)
    runner = StepRunner(reset_on_call=False)

    engine = Engine(config, agent=agent, env=env, runner=runner)

    train_logs = []
    checkpoint_count = 0
    for i in count():
        if agent.total_timestep >= config['train.timestep']:
            break
        train_logger = engine.train(i)
        train_logs.append(train_logger.logs)
        if i == 0 or (i+1) % config['log.freq'] == 0:
            train_logger.dump(keys=None, index=0, indent=0, border='-'*50)
        if agent.total_timestep >= int(config['train.timestep']*(checkpoint_count/(config['checkpoint.num'] - 1))):
            agent.checkpoint(logdir, i + 1)
            checkpoint_count += 1
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    return None

def test(n_tests, policy, render=False):
    scores = []
    actions = []
    for each_game in range(n_tests):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(env.spec.max_episode_steps):
            if render:
                env.render()
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                obs = torch.as_tensor(prev_obs, dtype=torch.float32).to(device)
                action = numpify(policy(obs).sample(), 'float')

            actions.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        scores.append(score)
    print('Average Score: {}'.format(sum(scores) / len(scores)))

if __name__ == '__main__':

    TRAIN = True
    RUN_ON_GPU = True
    LOG_DIR = "logs_experiment/".format(int(time.time()))

    device = torch.device("cpu")
    if RUN_ON_GPU:
        device = torch.device("cuda:0")

    if TRAIN:
        run_experiment(run=run,
                       config=config,
                       args=args,
                       seeds=[1770966829],
                       log_dir=LOG_DIR,
                       max_workers=os.cpu_count(),
                       chunksize=1,
                       use_gpu=RUN_ON_GPU,  # CPU a bit faster
                       gpu_ids=None)

    if not TRAIN:
        env = simreal.make('CustomPendulum-v0')
        agent = torch.load('logs/default/0/1770966829/model_agent_49.pt')
        test(n_tests=100, policy=agent, render=True)
        env.close()