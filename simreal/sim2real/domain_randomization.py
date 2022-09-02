from enum import Enum
import numpy as np
from scipy.stats import norm, truncnorm

def randomize_every(str_obj):
    if str_obj == 'episode':
        return RandomizeEvery.EPISODE
    elif str_obj == 'variation':
        return RandomizeEvery.VARIATION

def normal_dist_like(list):
    std = len(list)/2 * (norm.std()/norm.ppf(0.99))
    x = np.linspace(-len(list)/2, len(list)/2, len(list))
    y = norm.pdf(x, 0, std)
    if not sum(y) == 1.0:
        y[int(len(y)/2)] += 1.0-sum(y)
    return y

def get_truncated_normal(mean, std, low, up):
    return truncnorm((low - mean) / std, (up - mean) / std, loc=mean, scale=std)

class RandomizeEvery(Enum):
    EPISODE = 0
    VARIATION = 1

class ClassicDomain(object):
    def __init__(self, params):
        self.params = params

    def init_params(self):
        return self.params

class RandomDomainRandomization(object):
    def __init__(self, randomize_every, frequency, params, distribution):
        self.randomize_every = randomize_every,
        self.frequency = frequency
        self.params = params
        self.distribution = distribution
        self.last_randomized_count = 0
        self.count = 0

    def select(self, value):
        if self.distribution == 'uniform':
            if value[0] == 'list':
                selected_param = round(np.random.choice(value[1], p=None), 4)
            elif value[0] == 'min_max':
                min, max = value[1][0], value[1][1]
                selected_param = round(np.random.uniform(min, max), 4)
            elif value[0] == 'min_max_perc':
                min_perc, max_perc, initial = value[1][0], value[1][1], value[1][2]
                min, max = (100. - min_perc) / 100. * initial, (100. + min_perc) / 100. * initial
                selected_param = round(np.random.uniform(min, max), 4)

        elif self.distribution == 'normal':
            if value[0] == 'list':
                selected_param = round(np.random.choice(value[1], p=normal_dist_like(value[1])), 4)
            elif value[0] == 'min_max':
                min, max, initial = value[1][0], value[1][1], value[1][2]
                std = abs(max-min)/2 * (norm.std()/norm.ppf(0.99))
                truncated_normal = get_truncated_normal(mean=initial, std=std, low=min, up=max)
                selected_param = round(truncated_normal.rvs(), 4)
            elif value[0] == 'min_max_perc':
                min_perc, max_perc, initial = value[1][0], value[1][1], value[1][2]
                min, max = (100. - min_perc) / 100. * initial, (100. + min_perc) / 100. * initial
                std = abs(max - min) / 2 * (norm.std() / norm.ppf(0.99))
                truncated_normal = get_truncated_normal(mean=initial, std=std, low=min, up=max)
                selected_param = round(truncated_normal.rvs(), 4)
            elif value[0]== 'mean_std':
                mean, std = value[1][0], value[1][1]
                selected_param = round(np.random.normal(mean, std), 4)
        return selected_param

    def init_params(self):
        self.selected_params = {}
        for key in self.params:
            self.selected_params[key] = self.select(self.params[key])
        return self.selected_params

    def randomize(self):
        if self.count != self.last_randomized_count:
            for key in self.params:
                self.selected_params[key] = self.select(self.params[key])
            self.last_randomized_count = self.count
            print('Domain Randomized: Params = {}'.format(self.selected_params))
        return self.selected_params
