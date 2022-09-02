from gym.envs.registration import registry, register, make, spec

## -------- Custom Environments: Classic Control -------- ##

register(
    id='CustomCartPole-v1',
    entry_point='simreal.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CustomPendulum-v0',
    entry_point='simreal.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='CustomAcrobot-v1',
    entry_point='simreal.envs.classic_control:AcrobotEnv',
    reward_threshold=-100.0,
    max_episode_steps=500,
)
