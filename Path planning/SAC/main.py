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

from buffer import ReplayBuffer
from model import Actor, Critic, SAC
import Environment # <= for wind_map
from Environment import Env
import Environment_2
from Environment_2 import Env2

import matplotlib.pyplot as plt

'''
Hyper-parameter section
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AT_LR = 1e-4
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


goal_boundary = 10

RANDOM_SEED = 555
MAX_EPISODES = 20000
MAX_TIMESTEPS = 150
LOG_INTERVAL = 5


'''
Model directory
'''
ENV_NAME = "SAC"
directory = "./preTrained/{}".format(ENV_NAME) # save trained models
filename = "{}_{}".format(ENV_NAME, RANDOM_SEED)


log_f = open("SAC_ENV_T_reward", "w+")


def main():
    
    global wind_matrix

    # Initialize automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(action_dim)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = torch.exp(log_alpha).to(device)

    policy = SAC(AT_LR, CR_LR, state_dim, action_dim, max_action, gamma, target_entropy, log_alpha, alpha, tau, device)
    policy.load(directory, filename)
    buffer = ReplayBuffer(batch_size, max_size, device)
    
    # This part must be changed
    env = Env(goal_boundary)
    env_2 = Env_2(goal_boundary)

    for name, param in policy.actor.named_parameters():
        if 'bias' in name:
            param.data.zero_()

    for name, param in policy.critic_1.named_parameters():
        if 'bias' in name:
            param.data.zero_()

    for name, param in policy.critic_2.named_parameters():
        if 'bias' in name:
            param.data.zero_()

    if RANDOM_SEED:
        print("Random Seed: {}".format(RANDOM_SEED))
        #env.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    total_step = 0
    #log_f = open(filename,"w+")
    ep_list = []
    rwd_list = []
    epr_list = []

    # Add the full environment (scenario 1-6)
    for episode in range(1, MAX_EPISODES+1):
        if episode % 6 == 0:
            wind_matrix = Environment.wind_matrix1
        elif episode % 6 == 1:
            wind_matrix = Environment_2.wind_matrix1
        elif episode % 6 == 2:
            wind_matrix = Environment.wind_matrix2
        elif episode % 6 == 3:
            wind_matrix = Environment_2.wind_matrix2
        elif episode % 6 == 4:
            wind_matrix = Environment.wind_matrix3
        else:
            wind_matrix = Environment_2.wind_matrix3


        if episode % 2 == 0:
            state1, state2 = env.reset(episode, wind_matrix)
        else:
            state1, state2 = env_2.reset(episode, wind_matrix)
        
        #state1, state2 = env.reset(episode, wind_matrix)
        #print(env.agent, env.goal)

        for t in range(MAX_TIMESTEPS):

            mu, std = policy.actor(torch.FloatTensor(state1.reshape(1,-1)).to(device), torch.FloatTensor(state2.reshape(1,1,20,20,20)).to(device))
            action = policy.get_action(mu, std, max_action, action_bias)

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
            if episode % 2 == 0:
                next_state1, next_state2, reward, done = env.step(action, wind_matrix, t, MAX_TIMESTEPS)
            else:
                next_state1, next_state2, reward, done = env_2.step(action, wind_matrix, t, MAX_TIMESTEPS)

            # Add the transition (s1, s2, a, r, s1', s2', t)
            buffer.add((state1, state2, action, reward, next_state1, next_state2, float(done)))

            state1 = next_state1
            state2 = next_state2

            avg_reward += reward
            ep_reward += reward
            
            #time.sleep(1)

            if(len(buffer) >= batch_size):
                policy.update(buffer, batch_size, device)

            if done or t==(MAX_TIMESTEPS-1):
                total_step += t
                epr_list.append(ep_reward)
                #print("Number of zero weight: %d" % (policy.suma))
                #policy.suma = 0
                policy.save(directory, filename)
                break
        
        ep_reward = 0

        # Print average reward every interval:
        if episode % LOG_INTERVAL == 0:
            avg_reward = int(avg_reward / LOG_INTERVAL)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            print(epr_list)
            epr_list = []
            ep_list.append(episode)
            rwd_list.append(avg_reward)


            plt.figure(2)
            plt.plot(ep_list, rwd_list, 'r')
            plt.draw()
            plt.pause(0.0001)


            log_f.write('{}, {}\n'.format(episode, avg_reward))
            log_f.flush()

            avg_reward = 0



if __name__ == '__main__':
    main()