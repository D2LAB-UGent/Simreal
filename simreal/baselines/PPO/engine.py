import time
from itertools import count
import copy
import numpy as np

from simreal.rlbase.logger import Logger
from simreal import BaseEngine
from simreal.utils import describe
from simreal.utils import color_str


class Engine(BaseEngine):        
    def train(self, n=None, **kwargs):
        train_logs, eval_logs = [], []
        param_batch, param_episode = [], []
        checkpoint_count = 0

        for iteration in count():
            if self.agent.total_timestep >= self.config['train.timestep']:
                break
            self.agent.train()
            t0 = time.perf_counter()

            D = self.runner(self.agent, self.env, self.config['train.timestep_per_iter'], mode='train')
            out_agent = self.agent.learn(D)

            #add episode params to param_episode
            for traj in D[:len(D)-1]:
                param_episode = []
                for i in range(len(traj.timesteps)):
                    param_episode += [copy.deepcopy(traj.timesteps[i].params)]
                #add param_episode to the batch (which is cleared on log)
                param_batch += [param_episode]

            if self.config['env.normalize_reward']:
                std = np.sqrt(self.env.reward_moments.var + self.env.eps)
            else:
                std = 1

            logger = Logger()
            logger('train_iteration', iteration+1)
            logger('train_time', round(time.perf_counter() - self.agent.start_time, 1))
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            [logger(key, value) for key, value in out_agent.items()]
            logger('num_trajectories', len(D))
            logger('num_timesteps', sum([traj.T for traj in D]))
            logger('accumulated_trained_timesteps', self.agent.total_timestep)
            logger('return', describe([std*sum(traj.rewards) for traj in D], axis=-1, repr_indent=1, repr_prefix='\n'))

            E = [traj[-1].info['episode'] for traj in D if 'episode' in traj[-1].info]
            logger('online_return', describe([e['return'] for e in E], axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('online_horizon', describe([e['horizon'] for e in E], axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('running_return', describe(self.env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('running_horizon', describe(self.env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
            logger('param_batch', param_batch)
            train_logs.append(logger.logs)

            if iteration == 0 or (iteration + 1) % self.config['log.freq'] == 0:
                param_batch = []
                logger.dump(keys=None, index=0, indent=0, border='-' * 50)
            if self.agent.total_timestep >= int(self.config['train.timestep'] * (checkpoint_count / (self.config['checkpoint.num'] - 1))):
                self.agent.checkpoint(self.logdir, iteration + 1)
                checkpoint_count += 1
            if self.agent.total_timestep >= int(self.config['train.timestep'] * (len(eval_logs) / (self.config['eval.num'] - 1))):
                eval_logs.append(self.eval(n=len(eval_logs)))

        return train_logs, eval_logs
        
    def eval(self, n=None, **kwargs):
        t0 = time.perf_counter()

        if type(self.eval_runner).__name__ == 'StepRunner':
            D_eval = self.eval_runner(self.agent, self.eval_env, int(2.1*self.eval_env.spec.max_episode_steps), mode='eval') # at least 2 complete evals
            D_eval = D_eval[:len(D_eval)-1] #last trajectory not completed
        elif type(self.eval_runner).__name__ == 'EpisodeRunner':
            D_eval = self.eval_runner(self.agent, self.eval_env, 10, mode='eval') #10 times per eval

        if self.config['env.normalize_reward']:
            std = np.sqrt(self.eval_env.reward_moments.var + self.eval_env.eps)
        else:
            std = 1

        logger = Logger()
        logger('eval_iteration', n+1)
        logger('eval_time', round(time.perf_counter() - self.agent.start_time, 1))
        logger('num_seconds', round(time.perf_counter() - t0, 1))
        logger('accumulated_trained_timesteps', self.agent.total_timestep)
        logger('online_return', describe([std*sum(traj.rewards) for traj in D_eval], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('online_horizon', describe([traj.T for traj in D_eval], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_return', describe(self.eval_env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_horizon', describe(self.eval_env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger.dump(keys=None, index=0, indent=0, border=color_str('+' * 50, color='green'))

        return logger.logs
