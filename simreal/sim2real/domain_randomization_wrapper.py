import simreal
from simreal.sim2real.domain_randomization import randomize_every
from simreal.sim2real.domain_randomization import ClassicDomain
from simreal.sim2real.domain_randomization import RandomizeEvery, RandomDomainRandomization

class DomainRandomization(simreal.Wrapper):
    def __init__(self, env, args):
        self.env = env
        self.__dict__ = env.__dict__
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        #rewrite the overwritten time_limit wrapper
        self._max_episode_steps = self.env.spec.max_episode_steps
        self._elapsed_steps = None

        #define the environment randomizer
        if args['domain_randomization_type'] == 'none':
            self.Randomizer = ClassicDomain(params=args['params'])
        elif args['domain_randomization_type'] == 'random':
            self.Randomizer = RandomDomainRandomization(randomize_every=randomize_every(args['randomize_every']),
                                        frequency=args['frequency'], params=args['params'], distribution=args['distribution'])

        # initialize parameters
        self.params = self.Randomizer.init_params()

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"

        #critical
        self.env.__dict__['params'] = self.__dict__['params']

        observation, reward, done, info = self.env.step(action)

        #finish time_limit wrapper
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True

        #update parameters
        if type(self.Randomizer) != ClassicDomain:
            if self.Randomizer.count % self.Randomizer.frequency == 0 and self.Randomizer.count != 0:
                self.params = self.Randomizer.randomize()

            # Domain Randomization when randomizing on variations (per step)
            if self.Randomizer.randomize_every[0] == RandomizeEvery.VARIATION:
                self.Randomizer.count += 1
            # Domain Randomization when randomizing on episodes (per episode)
            if self.Randomizer.randomize_every[0] == RandomizeEvery.EPISODE:
                if done:
                    self.Randomizer.count += 1

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)