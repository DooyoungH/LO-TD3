import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torch.utils.data as data_utils

from torch.distributions import Normal

import numpy as np
import random
import pandas as pd
import time
import math

from buffer import ReplayBuffer
from model import Actor, Critic, SAC
import Environment # <= for wind_map
from Environment import Env

import pickle

'''
Hyper-parameter section
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AT_LR = 3e-4
CR_LR = 3e-4
gamma = 0.99
tau = 0.995

batch_size = 128
max_size = 5e5

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
MAX_EPISODES = 5000
MAX_TIMESTEPS = 80
LOG_INTERVAL = 5

'''
Model directory
'''
ENV_NAME = "SAC"
directory = "./preTrained/{}".format(ENV_NAME) # save trained models
filename = "{}_{}".format(ENV_NAME, RANDOM_SEED)


# For test and data generation

def get_action(mu, std, action_scale, action_bias):
    normal = Normal(mu, std)
    z = normal.mean
    # z = normal.rsample() <= reparameterization trick (mean + std * N(0,1))
    action = torch.tanh(z) * action_scale + action_bias

    return action.data.cpu().numpy().flatten()


def main():

    global wind_matrix

    if RANDOM_SEED:
        print("Random Seed: {}".format(RANDOM_SEED))
        env.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=lambda storage, loc: storage))
    buffer = ReplayBuffer(batch_size, max_size, device)
    env = Env(goal_boundary)

    demo = []

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    total_step = 0
    #log_f = open(filename,"w+")
    ep_list = []
    rwd_list = []

    for episode in range(1, MAX_EPISODES+1):
        if episode % 3 == 0:
            wind_matrix = Environment.wind_matrix1
        elif episode % 3 == 1:
            wind_matrix = Environment.wind_matrix2
        else:
            wind_matrix = Environment.wind_matrix3

        state1, state2 = env.reset(episode, wind_matrix)

        for t in range(MAX_TIMESTEPS):

            mu, std = actor(torch.FloatTensor(state1.reshape(1,-1)).to(device), torch.FloatTensor(state2.reshape(1,1,20,20,20)).to(device))
            action = get_action(mu, std, max_action, action_bias)

            action = np.array([action[0]*15, action[1]*15, action[2]*3], dtype=np.float64)
            action_sum = math.sqrt(((action[0])**2) + ((action[1])**2) + ((action[2])**2))

            # Action scaling (max velocity is 15 m/s)
            if action_sum > 15:
                action = (action/action_sum) * 15

            # Action cliping (max horizontal velocity is 15 m/s and vertical velocity is 3 m/s)
            action = np.clip(action, [lower_bound[0], lower_bound[1], lower_bound[2]], [upper_bound[0], upper_bound[1], upper_bound[2]])
            # Action normalization
            action = np.array([action[0]/15, action[1]/15, action[2]/3], dtype=np.float64)

            # Take action from Environment
            next_state1, next_state2, reward, done = env.step(action, wind_matrix, t, MAX_TIMESTEPS)

            # Add the transition (s1, s2, a, r, s1', s2', t)
            # buffer.add((state1, state2, action, reward, next_state1, next_state2, float(done)))
            transition = (state1, state2, action, reward, next_state1, next_state2, float(done))
            demo.append(transition)

            state1 = next_state1
            state2 = next_state2

            avg_reward += reward
            ep_reward += reward

            if done or t==(MAX_TIMESTEPS-1):
                total_step += t
                #policy.save(directory, filename)
                break

        ep_reward = 0

        # Print average reward every interval:
        if episode % LOG_INTERVAL == 0:
            avg_reward = int(avg_reward / LOG_INTERVAL)
            print("Episode: {}\tAverage Reward: {}\tStep: {}".format(episode, avg_reward, total_step))
            ep_list.append(episode)
            rwd_list.append(avg_reward)

            avg_reward = 0

    with open("SAC_Drone.pickle", "wb") as fw:
        pickle.dump(demo, fw)

if __name__ == '__main__':
    main()
