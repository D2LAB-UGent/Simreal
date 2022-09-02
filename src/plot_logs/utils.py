import numpy as np
import os
import torch
import pickle
import simreal
from simreal.utils import numpify
from simreal.wrappers import Noisy
import matplotlib.pyplot as plt

def graph_adr_stages(LOG_DIR_ACRO, seed):
    r = test_stages(LOG_DIR_ACRO, seed)

    fig = plt.plot()
    plt.plot(np.arange(len(r)), r)
    plt.xlabel('Iteration N')
    plt.ylabel('Return')
    plt.show()


def test_stages(LOG_DIR, seed):
    l = []
    for iteration in os.listdir(LOG_DIR):
        if iteration.startswith('PPO'):
            l += [int(iteration[len('PPO-'):])]

    env = simreal.make('CustomAcrobot-v1')
    env = Noisy(env, max_actuation_noise_perc=2, max_observation_noise_perc=2, max_offset_perc=0, friction=0.01)

    r_ar = []
    for i in range(np.max(l) + 1):
        iteration_dir = f'{LOG_DIR}/PPO-{i}/0/{seed}'
        m = 0
        for model in os.listdir(iteration_dir):
            if model[:12] == 'model_agent_':
                inter = model[12:]
                if int(inter[:len(inter) - 3]) > m:
                    m = int(inter[:len(inter) - 3])

        agent = torch.load('{}/model_agent_{}.pt'.format(iteration_dir, m))
        try:
            moments = pickle.load(open(f'{iteration_dir}/obs_moments_{m}.pth', 'rb'))
        except:
            moments = None

        r = test(n_tests=100, policy=agent, env=env, moments=moments, render=False)
        r_ar += [r]
    env.close()
    print(r_ar)
    return r_ar


def test(n_tests, policy, env, moments=None, render=True):
    scores = []
    actions = []

    RUN_ON_GPU = False
    device = torch.device("cpu")
    if RUN_ON_GPU:
        device = torch.device("cuda:0")

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
                if moments is not None:
                    (mean, std) = moments
                    prev_obs = (np.array(prev_obs) - mean) / std

                obs = torch.as_tensor(prev_obs, dtype=torch.float32).to(device)
                action = numpify(policy(obs).sample(), 'float')

            actions.append(action)
            # print(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        scores.append(score)
    print('Average Score: {}'.format(sum(scores) / len(scores)))
    return sum(scores) / len(scores)