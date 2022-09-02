import os, sys
import simreal
import time
import torch
import pickle
import numpy as np

import gym
from multiprocessing import Pool
from simreal import DomainRandomization
from simreal import StepRunner, EpisodeRunner, Logger
from simreal.utils import describe
from simreal.utils import pickle_dump, set_global_seeds, numpify, tensorify, CloudpickleWrapper
from simreal.rlbase import Config, Grid, run_experiment
from simreal.wrappers import RecordEpisodeStatistics, NormalizeObservation, NormalizeReward
from simreal.wrappers import TimeStepEnv, Noisy

from simreal.baselines.PPO.agent import Agent
from simreal.baselines.PPO.engine import Engine

from simreal import CEM
from simreal.utils.Visualizations import plot_cem

import matplotlib
matplotlib.use('TkAgg')


class ADR(object):
    def __init__(self, iter_max, config_PPO, args_0, config_CEM, logdir, run_on_gpu, seed):
        self.environment = config_PPO.items['env.id'][0]
        self.iter_max = iter_max
        self.config_PPO = config_PPO
        self.config_CEM = config_CEM
        self.args_0 = args_0
        self.args = args_0
        self.logdir = logdir
        self.run_on_gpu = run_on_gpu
        self.init_train_steps = self.config_PPO.items['train.timestep']

        self.seed = seed
        self.iter = 0

        # create devices
        self.device_CEM = torch.device("cpu")
        if run_on_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.real_env = self.make_env(args=args_0, mode='eval', to_train=False)
        self.eval_env = self.make_env(args=args_0, mode='eval', to_train=True)

        # initialize agent, moments, means and stds
        self.agent = None
        self.PPO_agent = None
        self.moments = None
        self.means = config_CEM.items['train.mu0']
        self.stds = config_CEM.items['train.std0']
        self.iter_eval = []
        self.update_eps_mean = 0.1       # clip difference in phi across updates
        self.update_eps_std = 0.5

    # Adaptive domain randomization - framework
    def run(self):
        while True:
            LOG_DIR_PPO = "{}/PPO-{}".format(self.logdir, self.iter)
            LOG_DIR_CEM = "{}/CEM-{}".format(self.logdir, self.iter)

            self.start_PPO(args=self.args, log_dir=LOG_DIR_PPO)
            self.extract_policy(LOG_DIR_PPO)
            self.evaluate_iteration()

            if self.iter > self.iter_max:
                break

            self.start_CEM(log_dir=LOG_DIR_CEM)
            self.extract_cem(LOG_DIR_CEM)

            self.update_phi()
            self.iter += 1
        return None

    # create environments
    def make_env(self, args, mode, to_train=False):
        assert mode in ['train', 'eval']

        env = simreal.make(self.environment)
        env.seed(self.seed[0])
        env.observation_space.seed(self.seed[0])
        env.action_space.seed(self.seed[0])

        if to_train:
            if self.config_PPO.items['env.clip_action'] and isinstance(env.action_space, gym.spaces.Box):
                env = gym.wrappers.ClipAction(env)

        if mode == 'train':
            env = DomainRandomization(env, args)
            env = RecordEpisodeStatistics(env, deque_size=100)
        elif mode == 'eval':
            env = Noisy(env, max_actuation_noise_perc=2, max_observation_noise_perc=2, max_offset_perc=0, friction=0.01)
            env = RecordEpisodeStatistics(env, deque_size=100)

        if to_train:
            if self.config_PPO.items['env.normalize_obs']:
                env = NormalizeObservation(env, clip=5.)
            if self.config_PPO.items['env.normalize_reward']:
                env = NormalizeReward(env, clip=10., gamma=self.config_PPO.items['agent.gamma'])
            env = TimeStepEnv(env)
        return env

    # Setup PPO
    def run_PPO(self, config, seed, device, logdir, args):
        print('PPO running on: {}'.format(device))
        set_global_seeds(seed)

        env = self.make_env(args, 'train', to_train=True)
        # eval_env = self.make_env(args, 'eval', to_train=True)
        if self.agent != None:
            agent = self.PPO_agent
            config['train.timestep'] = self.init_train_steps + agent.total_timestep
        else:
            agent = Agent(config, env, self.device)

        # agent = Agent(config, env, self.device)

        runner = StepRunner(reset_on_call=True)
        eval_runner = StepRunner(reset_on_call=True)

        engine = Engine(config, agent=agent, env=env, eval_env=self.eval_env, runner=runner, eval_runner=eval_runner, logdir=logdir)

        train_logs, eval_logs = engine.train()
        pickle_dump(obj=train_logs, f=logdir / 'train_logs', ext='.pkl')
        pickle_dump(obj=eval_logs, f=logdir / 'eval_logs', ext='.pkl')
        return None

    def start_PPO(self, args, log_dir):
        run_experiment(run=self.run_PPO,
                       config=self.config_PPO,
                       args=args,
                       seeds=self.seed,
                       log_dir=log_dir,
                       max_workers=None,        # None: Mandatory during adaptive domain randomization!
                       chunksize=1,
                       use_gpu=self.run_on_gpu,
                       gpu_ids=None)

    # Extract policy from PPO logdir
    def extract_policy(self, LOG_DIR_PPO):
        dir = '{}/0/{}'.format(LOG_DIR_PPO, self.seed[0])
        ids = []
        for el in os.listdir(dir):
            if el.startswith('model_agent_'):
                id = int(el.replace('model_agent_', '').replace('.pt', ''))
                ids += [id]

        self.agent = torch.load('{}/model_agent_{}.pt'.format(dir, np.max(ids)))
        try:
            self.moments = pickle.load(open(f'{dir}/obs_moments_{np.max(ids)}.pth', 'rb'))
        except:
            self.moments = None

        self.PPO_agent = pickle.load(open('{}/PPO_agent_{}.pkl'.format(dir, np.max(ids)), 'rb'))

    def evaluate_iteration(self):
        print('Evaluating ADR iteration ... ')
        n_tests = 100
        scores = []
        for each_game in range(n_tests):
            score = 0
            prev_obs = []
            self.real_env.reset()
            for _ in range(self.real_env.spec.max_episode_steps):
                if len(prev_obs) == 0:
                    action = self.real_env.action_space.sample()
                else:
                    if self.moments is not None:
                        (mean, std) = self.moments
                        prev_obs = (np.array(prev_obs) - mean) / std

                    obs = torch.as_tensor(prev_obs, dtype=torch.float32).to(self.device)
                    action = numpify(self.agent(obs).sample(), 'float')

                new_observation, reward, done, info = self.real_env.step(action)
                prev_obs = new_observation
                score += reward
                if done:
                    break
            scores.append(score)
        self.iter_eval += [sum(scores) / len(scores)]
        print('_'*100)
        print('ITERATION COMPLETE: {}'.format(self.iter_eval))
        print('_'*100)

    # Extract CEM results
    def extract_cem(self, LOG_DIR_CEM):
        dir = '{}/0/{}'.format(LOG_DIR_CEM, self.seed[0])
        for el in os.listdir(dir):
            if el.startswith('train_logs'):
                with open('{}/{}'.format(dir, el), "rb") as input_file:
                    pkl = pickle.load(input_file)

        # extract means and stds from the last generation
        means = np.array(pkl[-1]['means'][0])
        stds = np.array(pkl[-1]['stds'][0])

        # clip update difference
        delta_means = np.array(means) - np.array(self.means)
        delta_stds = np.array(stds) - np.array(self.stds)
        delta_means_max, delta_stds_max = np.abs(self.update_eps_mean*np.array(self.means)), \
                                          np.abs(self.update_eps_std*np.array(self.stds))

        for i in range(len(means)):
            self.means[i] += np.clip(delta_means[i], -delta_means_max[i], delta_means_max[i])
            self.stds[i] += np.clip(delta_stds[i], -delta_stds_max[i], delta_stds_max[i])

    # Setup CEM
    def get_eval_acts_obs(self, real_env):
        self.agent = self.agent.to(self.device_CEM)
        score = 0
        eval_trajectory = []
        prev_obs = real_env.reset()
        prev_obs_real = prev_obs
        real_env.reset()

        for _ in range(real_env.spec.max_episode_steps):
            if len(eval_trajectory) == 0:
                action = numpify(np.zeros_like(real_env.action_space.sample()), 'float')
            else:
                prev_obs_real = prev_obs
                if self.moments is not None:
                    (mean, std) = self.moments
                    prev_obs = (np.array(prev_obs) - mean)/std

                obs = torch.as_tensor(prev_obs, dtype=torch.float32).to(self.device_CEM)
                action = numpify(self.agent(obs).sample(), 'float')

            new_observation, reward, done, info = real_env.step(action)

            eval_trajectory.append([(prev_obs_real, action), new_observation])
            prev_obs = new_observation
            score += reward
            if done:
                break
        return score, eval_trajectory

    def get_sim_obs(self, env, param, eval_trajectory):
        sim_trajectory = []
        for i in range(len(eval_trajectory)):
            if self.environment == 'CustomCartPole-v1':
                args_param = {'domain_randomization_type': 'none',
                            'params': {'masscart': param[0], 'masspole': param[1], 'length': param[2]}}
            elif self.environment == 'CustomPendulum-v0':
                args_param = {'domain_randomization_type': 'none',
                            'params': {'m': param[0], 'l': param[1]}}
            elif self.environment == 'CustomAcrobot-v1':
                args_param = {'domain_randomization_type': 'none',
                            'params': {'l1': param[0], 'l2': param[1], 'm1': param[2], 'm2': param[3]}}
            env = DomainRandomization(env, args_param)

            # set sim_env to eval state
            env.reset()  # doesn't reset parameters only state, required for calling step()
            state = eval_trajectory[i][0][0]
            env.state = self.obs_to_state(state)

            # apply action on sim_env and record sim_obs
            action = eval_trajectory[i][0][1]
            sim_ob, reward, done, info = env.step(action)

            sim_trajectory.append([(state, action), sim_ob])
        env.close()
        return sim_trajectory

    def obs_to_state(self, obs):
        if self.environment == 'CustomCartPole-v1':
            s = np.array(obs)
        elif self.environment == 'CustomPendulum-v0':
            s1 = np.arctan2(obs[1], obs[0])
            s = np.array([s1, obs[2]])
        elif self.environment == 'CustomAcrobot-v1':
            s1 = np.arctan2(obs[1], obs[0])
            s2 = np.arctan2(obs[3], obs[2])
            s = np.array([s1, s2, obs[4], obs[5]])
        return s

    def clip_angle_difference(self, difference):
        if self.environment == 'CustomCartPole-v1':
            angle_idx = [2]
        elif self.environment == 'CustomPendulum-v0':
            angle_idx = [0]
        elif self.environment == 'CustomAcrobot-v1':
            angle_idx = [0, 1]

        for idx in angle_idx:
            if difference[idx]>np.pi:
                difference[idx] = 2*np.pi-difference[idx]
            
        return difference

    def fitness(self, data):
        torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
        config, seed, device_CEM, param, eval_trajectory = data

        #Get observations in sim env
        env = self.make_env(args=self.args_0, mode='train', to_train=False)
        sim_trajectory = self.get_sim_obs(env, param, eval_trajectory)

        D = 0
        check = []
        if self.environment == 'CustomCartPole-v1':
            weights = np.array([1, 1, 1, 1])
        elif self.environment == 'CustomPendulum-v0':
            weights = np.array([1, 1])
        elif self.environment == 'CustomAcrobot-v1':
            weights = np.array([.286, .10, .010, .0667])

        #Discrepancy function
        for i in range(len(sim_trajectory)):
            sim_obs = self.obs_to_state(sim_trajectory[i][1])
            eval_obs = self.obs_to_state(eval_trajectory[i][1])
            difference = abs(sim_obs-eval_obs)

            #clip difference in angles
            difference = self.clip_angle_difference(difference)

            check += [np.array(difference)]

            #weighted sum
            D += np.sum(difference*weights)
        return D/len(sim_trajectory)

    def run_CEM(self, config, seed, device_CEM, logdir, args):
        print('Using global real-env: {}'.format(self.real_env.spec))
        print('CEM running on: {}'.format(device_CEM))
        set_global_seeds(seed)
        torch.set_num_threads(1)  # VERY IMPORTANT TO AVOID GETTING STUCK
        print('Initializing...')

        es = CEM(config['train.mu0'], config['train.std0'],
                 {'popsize': config['train.popsize'],
                  'seed': seed,
                  'elite_amount': config['train.elite_amount']})
        train_logs = []
        params_logs, elite_logs, mean_logs, std_logs, Fs_logs = [], [], [], [], []
        with Pool(processes=config['train.popsize']//config['train.worker_chunksize']) as pool:
            print('Finish initialization. Training starts...')
            for generation in range(config['train.generations']):

                solutions = es.ask()

                eval_return, eval_trajectory = self.get_eval_acts_obs(self.real_env)

                data = [(config, seed, self.device_CEM, solution, eval_trajectory) for solution in solutions]

                out = pool.map(CloudpickleWrapper(self.fitness), data, chunksize=config['train.worker_chunksize'])
                Fs = np.array(out)
                es.tell(solutions, Fs)

                # Logs for visualizations
                params_logs += [solutions]
                elite_logs += [es.used_elites]
                mean_logs += [es.used_elites.mean(axis=0)]
                std_logs += [es.used_elites.std(axis=0)]
                Fs_logs += [Fs]

                logger = Logger()
                logger('generation', generation+1)
                logger('Fitness', describe(Fs, axis=-1, repr_indent=1, repr_prefix='\n'))
                logger('fbest', es.result.fbest)
                logger('means', es.used_elites.mean(axis=0))
                logger('stds', es.used_elites.std(axis=0))
                train_logs.append(logger.logs)
                if generation == 0 or (generation+1) % config['log.freq'] == 0:
                    logger.dump(keys=None, index=0, indent=0, border='-'*50)

        visualize = False
        if visualize:
            plot_cem(params_logs, elite_logs, mean_logs, std_logs, Fs_logs, heatmap=True)

        pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
        return None

    def start_CEM(self, log_dir):
        run_experiment(run=self.run_CEM,
                       config=self.config_CEM,
                       args=None,
                       seeds=self.seed,
                       log_dir=log_dir,
                       max_workers=None,         # None: Mandatory during adaptive domain randomization!
                       chunksize=1,
                       use_gpu=False,
                       gpu_ids=None)

    def update_phi(self):
        # update phi inside env args
        for i, key in enumerate(self.args['params'].keys()):
            self.args['params'][key] = ('mean_std', [self.means[i], self.stds[i]])

        # update phi inside CEM
        self.config_CEM.items['train.mu0'] = self.means
        self.config_CEM.items['train.std0'] = self.stds
