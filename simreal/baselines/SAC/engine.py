import time
from itertools import count
import copy

import torch
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
            t0 = time.perf_counter()
            
            if iteration < self.config['replay.init_trial']:
                D = self.runner(self.random_agent, self.env, 1)
            else:
                D = self.runner(self.agent, self.env, 1, mode='train')

            [traj] = D
            self.replay.add(traj)

            # Number of gradient updates = collected episode length
            out_agent = self.agent.learn(D=None, replay=self.replay, T=traj.T)

            #add episode params to param_episode
            param_episode = []
            for i in range(len(traj.timesteps)):
                param_episode += [copy.deepcopy(traj.timesteps[i].params)]
            #add param_episode to the batch (which is cleared on log)
            param_batch += [param_episode]

            logger = Logger()
            logger('train_iteration', iteration+1)
            logger('train_time', round(time.perf_counter() - self.agent.start_time, 1))
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            [logger(key, value) for key, value in out_agent.items()]
            logger('episode_return', sum(traj.rewards))
            logger('episode_horizon', traj.T)
            logger('accumulated_trained_timesteps', self.agent.total_timestep)
            logger('param_batch', param_batch)
            train_logs.append(logger.logs)
            if iteration == 0 or (iteration+1) % self.config['log.freq'] == 0:
                param_batch = []
                logger.dump(keys=None, index=0, indent=0, border='-'*50)
            if self.agent.total_timestep >= int(self.config['train.timestep']*(checkpoint_count/(self.config['checkpoint.num'] - 1))):
                self.agent.checkpoint(self.logdir, iteration + 1)
                checkpoint_count += 1
                
            if self.agent.total_timestep >= int(self.config['train.timestep']*(len(eval_logs)/(self.config['eval.num'] - 1))):
                eval_logs.append(self.eval(n=len(eval_logs)))
        return train_logs, eval_logs

    def eval(self, n=None, **kwargs):
        t0 = time.perf_counter()
        with torch.no_grad():
            if type(self.runner).__name__ == 'StepRunner':
                D_eval = self.runner(self.agent, self.eval_env, int(2.1 * self.eval_env.spec.max_episode_steps),
                                     mode='eval')  # at least 2 complete evals
                D_eval = D_eval[:len(D_eval) - 1]  # last trajectory not completed
            elif type(self.runner).__name__ == 'EpisodeRunner':
                D_eval = self.runner(self.agent, self.eval_env, 10, mode='eval')  # 10 times per eval

        logger = Logger()
        logger('eval_iteration', n+1)
        logger('eval_time', round(time.perf_counter() - self.agent.start_time, 1))
        logger('num_seconds', round(time.perf_counter() - t0, 1))
        logger('accumulated_trained_timesteps', self.agent.total_timestep)
        logger('online_return', describe([sum(traj.rewards) for traj in D_eval], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('online_horizon', describe([traj.T for traj in D_eval], axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_return', describe(self.eval_env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_horizon', describe(self.eval_env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger.dump(keys=None, index=0, indent=0, border=color_str('+'*50, color='green'))
        return logger.logs
