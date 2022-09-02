import simreal
import numpy as np
import copy

class Noisy(simreal.Wrapper):
    def __init__(self, env, max_actuation_noise_perc=0, max_observation_noise_perc=0, max_offset_perc=0, friction=0.1):
        self.env = env
        self.__dict__ = env.__dict__

        self.max_actuation_noise_perc = max_actuation_noise_perc            #example: 2%
        self.max_observation_noise_perc = max_observation_noise_perc        #example: 2%
        self.max_offset_perc = max_offset_perc                              #example: 5%
        self.friction = friction

        #remember initial parameters
        self.IP = copy.deepcopy(self.params)

        #parameter offset
        if self.max_offset_perc > 0:
            for key in self.params:
                change_per = np.random.uniform(-self.max_offset_perc, self.max_offset_perc)
                self.params[key] = round(self.IP[key] + ((change_per/100.) * self.IP[key]), 4)

    def step(self, action):
        #noisy actuation
        if self.max_actuation_noise_perc > 0:
            change_per = np.random.uniform(-self.max_actuation_noise_perc, self.max_actuation_noise_perc, size=np.shape(action))
            action += (change_per/100.) * action

        #critical
        try:
            self.env.__dict__['params'] = self.__dict__['params']
        except:
            pass

        observation, reward, done, info = self.env.step(action)

        #noisy observation
        if self.max_observation_noise_perc > 0:
            change_per = np.random.uniform(-self.max_observation_noise_perc, self.max_observation_noise_perc, size=np.shape(observation))
            observation += (change_per/100.) * observation

        return observation, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)