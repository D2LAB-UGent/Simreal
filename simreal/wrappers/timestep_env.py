import simreal

from simreal.rlbase.data import StepType
from simreal.rlbase.data import TimeStep


class TimeStepEnv(simreal.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        params = self.env.params
        # print('PARAMETERS in timestep', self.env.params)
        step_type = StepType.LAST if done else StepType.MID
        timestep = TimeStep(step_type=step_type, observation=observation, reward=reward, done=done, info=info, params=params)
        return timestep

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        params = self.env.params
        return TimeStep(StepType.FIRST, observation=observation, reward=None, done=None, info=None, params=params)
