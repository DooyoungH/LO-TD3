import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torch.utils.data as data_utils

import numpy as np
import random
import pandas as pd
import time
import math
import pickle

from TD3_buffer import ReplayBuffer
from TD3BC import Actor, Critic, TD3_BC
import Environment # <= for wind_map
from Environment import Env


'''
Hyper-parameter section
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AT_LR = 1e-4
CR_LR = 3e-4
gamma = 0.99
tau = 0.995
policy_noise = 0.2
noise_clip = 0.5
policy_freq = int(2)
alpha = 2.5

batch_size = 128
max_size = 5e5
normalize = True


state_dim = 12
print("Size of State Space -> {}".format(state_dim))
action_dim = 3
print("Size of Action Space -> {}".format(action_dim))
max_action = 1
action_bias = 0
upper_bound = [15, 15, 3]
print("Max Value of Action -> {}".format(upper_bound))
lower_bound = [-15, -15, -3]
print("Min Value of Action -> {}".format(lower_bound))

goal_boundary = 5

RANDOM_SEED = 0
#MAX_EPISODES = 5000
MAX_TIMESTEPS = 100000
TIMELIMITS = 80
LOG_INTERVAL = 5

eval_freq = int(1e3)

'''
Save directory
'''
ENV_NAME = "TD3BC"
directory = "./preTrained/{}".format(ENV_NAME) # save trained models
filename = "{}_{}".format(ENV_NAME, RANDOM_SEED)

def eval_policy(policy, env_name, seed, mean_1, std_1, mean_2, std_2, seed_offset=100, eval_episode=10):
    eval_env = Env(goal_boundary)
    #eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for episode in range(eval_episode):

        if episode % 3 == 0:
            wind_matrix = Environment.wind_matrix1
        elif episode % 3 == 1:
            wind_matrix = Environment.wind_matrix2
        else:
            wind_matrix = Environment.wind_matrix3
        
        state1, state2 = eval_env.reset(episode, wind_matrix)

        for t in range(TIMELIMITS):
            state1 = (np.array(state1).reshape(1, -1) - mean_1) / std_1
            state2 = (np.array(state2).reshape(1, 1, 20, 20, 20) - mean_2) /std_2

            action = policy.select_action(state1, state2, device)

            action = np.array([action[0]*15, action[1]*15, action[2]*3], dtype=np.float64)
            action_sum = math.sqrt(((action[0])**2) + ((action[1])**2) + ((action[2])**2))

            # Action scaling (max velocity is 15 m/s)
            if action_sum > 15:
                action = (action/action_sum) * 15

            # Action cliping (max horizontal velocity is 15 m/s and vertical velocity is 3 m/s)
            action = np.clip(action, [lower_bound[0], lower_bound[1], lower_bound[2]], [upper_bound[0], upper_bound[1], upper_bound[2]])
            #print(eval_env.agent, action)

            # Action normalization
            action = np.array([action[0]/15, action[1]/15, action[2]/3], dtype=np.float64)

            # Take action from Environment
            next_state1, next_state2, reward, done = eval_env.step(action, wind_matrix, t, TIMELIMITS)

            state1 = next_state1
            state2 = next_state2
            avg_reward += reward

            if done or t==(TIMELIMITS-1):
                #print(avg_reward, episode)
                break

    avg_reward = avg_reward / eval_episode

    return avg_reward


def main():

    global wind_matrix

    policy = TD3_BC(AT_LR, CR_LR, state_dim, action_dim, max_action, gamma, tau, policy_noise, noise_clip, policy_freq, alpha, device)
    #policy.load(directory, filename)
    buffer = ReplayBuffer(batch_size, max_size, state_dim, state_dim, action_dim, device)
    env = Env(goal_boundary)

    if RANDOM_SEED:
        print("Random Seed: {}".format(RANDOM_SEED))
        env.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # logging variables:
    ep_reward = 0
    total_step = 0
    #log_f = open(filename,"w+")
    ep_list = []
    rwd_list = []

    demo_path = "SAC_Drone.pickle"
    with open(demo_path, "rb") as f:
        demo = pickle.load(f)

    buffer.convert_ARRAY(demo)

    if normalize:
        mean_1, std_1, mean_2, std_2 = buffer.normalize_states()
    else:
        mean_1, std_1, mean_2, std_2 = 0., 0., 0., 0.

    evaluations = []

    for t in range(MAX_TIMESTEPS):
        policy.update(buffer, batch_size, device)
        avg_rwd = 0

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            avg_rwd = eval_policy(policy, ENV_NAME, RANDOM_SEED, mean_1, std_1, mean_2, std_2)
            print(f"Time step: {t + 1}, Average reward: {avg_rwd}")
            avg_rwd = "{}, {} \n".format(t+1, avg_rwd)

if __name__ == '__main__':
    main()