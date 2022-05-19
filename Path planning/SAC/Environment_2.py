# import the modules

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
import random
import pandas as pd
import time
import math
import csv
import copy
import matplotlib.pyplot as plt

from collections import deque, namedtuple
from typing import Deque, Dict, List, Tuple
from datetime import datetime
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split

'''
Confirm the GPU setting
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


'''
Load windmap files

This section loads windmap files configured by CFD.
There are two types of windmap follwing papers.
One set of windmaps contains wind direction effect.
On the other hand, another set of windmaps contains wind speed effect.
'''

wind_1 = open('./weights/windmap_N.csv', 'r', encoding='utf-8')
wind_1 = np.array(list(csv.reader(wind_1)))
wind_1 = np.reshape(wind_1, (20, 2, 30, 20))

wind_2 = open('./weights/windmap_NE.csv', 'r', encoding='utf-8')
wind_2 = np.array(list(csv.reader(wind_2)))
wind_2 = np.reshape(wind_2, (20, 2, 30, 20))
#print(wind_2.shape)

wind_3 = open('./weights/windmap_SW.csv', 'r', encoding='utf-8')
wind_3 = np.array(list(csv.reader(wind_3)))
wind_3 = np.reshape(wind_3, (20, 2, 30, 20))
#print(wind_3.shape)

wind_matrix1 = wind_1.astype(float)
wind_matrix2 = wind_2.astype(float)
wind_matrix3 = wind_3.astype(float)


'''
Load Drone power model

This section loads the drone power consumption model based on DJI Matrice 600.
The model contains drone's height, velocity, acceleration, payload, and wind affection as inputs.
The model has 10% errors compeared with actual flight data.
'''

class DNN(nn.Module):
    def __init__(self, input_dim, d_hidden, d_layer):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(input_dim, d_hidden)
        for i in range(d_layer):
            filename="l{}".format(i+1)
            self.filename = nn.Linear(d_hidden, d_hidden)
        self.l_out = nn.Linear(d_hidden, 1)
    
    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        for i in range(d_layer):
            filename="l{}".format(i+1)
            x = F.relu(self.filename(x))
        out = self.l_out(x)
        return out

# Hyperparameters

d_hidden = 30
d_layer = 50
input_dim = 9 #len(train_X[1])
criterion = nn.MSELoss()
learning_rate = 0.0002

model = DNN(input_dim, d_hidden, d_layer).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

directory = "./model"
filename = "DNN"
name = filename + '_solved'

'''
torch.save(model.state_dict(), '%s_model.pth' % (name))
print("model is saved")
'''

model.load_state_dict(torch.load('%s_model.pth' % (name), map_location=lambda storage, loc: storage))
print("model is loaded")




'''
Configuration Virtual Environment

real-time simulators are correct and simple, but they are quite heavy to using the ML training.
This step configure simple 3D virtual environment congifured by the graph.
This environment contains 3D obstacles and wind information at each grid to motivate the real-environment.
'''

# Obstacle information
N = [60, 60, 60,180, 180, 220, 200, 280, 280, 260]
E = [40, 180, 300, 60, 200, 320, 340, 20, 140, 300]
D = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

dN = [60, 60, 60, 60, 60, 20, 100, 60, 80, 80]
dE = [60, 60, 60, 60, 80, 40, 20, 60, 100, 60]
dD = [44, 32, 42, 28, 44, 38, 30, 32, 36, 38]

# figure setting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.set_xlim(0, 400)
ax.set_ylim(0, 400)
ax.set_zlim(0, 60)
ax.bar3d(N, E, D, dN, dE, dD)


'''
Agent logging start
'''

log_f = open("SAC_agent_status","w+")


class Env:

    def __init__(self, goal_boundary):
        # 400 x 400 x 60 (m)
        self.env_matrix = np.zeros((400, 400, 60))
        self.agent = np.array([0, 0, 0])
        self.goal = np.array([400, 400, 60])
        # range of 3D convolution (Lidar)
        self.camera_size = [20, 20, 20]
        self.goal_boundary = goal_boundary
        self.beta = 0
        self.step_cnt = -2
        self.max_step = 1000
        # save previous motion to derive the acc.
        self.prev_action = [0.0, 0.0, 0.0]
        self.scatter_array = []
        self.energy = 0

        self.step_reward = 0
        
        # for reward calculation
        self.toward_distance = 0

        # for normalization
        data = pd.read_csv('training_data.csv', header = None)

        train_Cols = [0, 1, 2, 3, 4, 5, 6, 8, 9]
        target_Cols = [11]

        tr_data = data[train_Cols]
        ta_data = data[target_Cols]

        self.d_max = tr_data.max()
        self.d_min = tr_data.min()

        # Build the obstacles in the environment
        # z-axis => x-axis => y=>axis
        for i in range(len(N)):
            for z in range(D[i], dD[i]):
                for x in range(N[i], N[i]+dN[i]):
                    for y in range(E[i], E[i]+dE[i]):
                        self.env_matrix[x][y][z] = 1

        # Collision check of agent
        while True:
            self.agent = np.array([random.randint(0, 400-1), random.randint(0, 400-1), random.randint(0, 60-1)])
            if self.env_matrix[self.agent[0]][self.agent[1]][self.agent[2]] != 1:
                break

        # Collision check of goal
        while True:
            self.goal = np.array([random.randint(0, 400-1), random.randint(0, 400-1), random.randint(0, 60-1)])
            if self.env_matrix[self.agent[0]][self.agent[1]][self.agent[2]] != 1:
                break

        # Wind information initialization
        agent_int = self.agent.astype(int)
        agent_for_wind = np.clip(agent_int, [0, 0, 0], [400-1, 400-1, 60-1])

        self.wind_dir = 0
        self.wind_vel = 0

        # Euclidean distance initialzation for reward
        self.distance = math.sqrt(((self.goal[0]-self.agent[0])**2) + ((self.goal[1]-self.agent[1])**2) + ((self.goal[2]-self.agent[2])**2))
        self.next_distance = 0

        self.min_distance = self.distance


    def get_3D_state(self):
        # initailize the recognition range of agent
        temp_matrix = np.zeros((1, self.camera_size[0], self.camera_size[1], self.camera_size[2]), dtype=np.float64)
        # [batch x recognize range_x x recognize range_y x recognize range_z]

        # location of agent (required casting)
        x_, y_, z_ = int(self.agent[0]), int(self.agent[1]), int(self.agent[2])
        # recognization range of agent (required casting)
        x__, y__, z__ = int(self.camera_size[0]/2), int(self.camera_size[1]/2), int(self.camera_size[2]/2)

        #print(x_, y_, z_)
        #print(x__, y__, z__)

        # Check the recognization range that overcomes the environment
        # agent location range 10~390, 10~390, 10~50
        if x_ < x__:
            x_ = x__
        if y_ < y__:
            y_ = y__
        if z_ < z__:
            z_ = z__
        if x_ > 400 - x__:
            x_ = 400 - x__
        if y_ > 400 - y__:
            y_ = 400 - y__
        if z_ > 60 - z__:
            z_ = 60 - z__

        for x in range(self.camera_size[0]):
            for y in range(self.camera_size[1]):
                for z in range(self.camera_size[2]):
                    temp_matrix[0][x][y][z] = self.env_matrix[x_-x__+x][y_-y__+y][z_-z__+z]
        
        # temp_matrix : Information of the drone arround of 20 m.
        return temp_matrix


    def norm(self, x, d_max, d_min):
        '''
        Must confirm constant values from previous code
        d_max = 0, d_min = 0
        '''
        return (x - d_min) / (d_max - d_min)


    def step(self, action, wind_matrix, step, max_step):
        # action : drone velocity range, horizontal : -15 ~ 15 m/s, vertical : -3 ~ 3 m/s
        # saved action : nomalized
        # action in step : original values
        action = np.array([action[0]*15, action[1]*15, action[2]*3], dtype=np.float64)
        # 1 Hz, drone location following action
        self.agent = self.agent + action

        agent_int = self.agent.astype(int)
        # wind position indicator
        agent_for_wind = np.clip(agent_int, [0, 0, 0], [400-1, 400-1, 60-1])

        '''
        wind matrix [x][vel:0, dir:1][z][y]
        wind matrix derived by CFD has grid resolution 400 x 400 x 60 (m).
        So, I determine the wind information to divide the location of agent with grid size (20 x 20 x 2).
        '''
        self.wind_dir = wind_matrix[int(agent_for_wind[0]/20)][1][int(agent_for_wind[2]/2)][int(agent_for_wind[1]/20)]
        self.wind_vel = wind_matrix[int(agent_for_wind[0]/20)][0][int(agent_for_wind[2]/2)][int(agent_for_wind[1]/20)]
        # Euclidean distance for reward function.
        self.next_distance = math.sqrt(((self.goal[0]-self.agent[0])**2) + ((self.goal[1]-self.agent[1])**2) + ((self.goal[2]-self.agent[2])**2))

        # logging the agent information
        log_f.write('{}\n'.format(self.agent))
        log_f.flush()

        # input information (State t+1)
        # next_state contains their normalization.
        # State information : drone arround information  [batch x 20 x 20 x 30]
        #                   : each location information (drone, action, goal, wind, etc..)
        state_3D = self.get_3D_state()
        next_state = [self.agent[0]/400, self.agent[1]/400, self.agent[2]/60, action[0]/15, action[1]/15, action[2]/3, self.goal[0]/400, self.goal[1]/400, self.goal[2]/60, self.wind_dir/360, self.wind_vel/10]

        '''
        Checksum termianl conditions
        Drones overcome the map, collide with obstcles, and approach the goal.
        '''

        if self.agent[0]<0 or self.agent[0]>=400 or self.agent[1]<0 or self.agent[1]>=400 or self.agent[2]<0 or self.agent[2]>=60:
            print("Check: Out of scope")
            reward = -100
            done = True
            success = False
            
            return np.array(next_state, dtype=np.float64), state_3D, reward, done

        elif self.env_matrix[agent_int[0]][agent_int[1]][agent_int[2]] == 1:
            print("Check: Collide with obstacles")
            reward = -100
            done = True
            success = False

            return np.array(next_state, dtype=np.float64), state_3D, reward, done

        elif self.agent[0]<self.goal[0]+self.goal_boundary and self.agent[0]>self.goal[0]-self.goal_boundary and self.agent[1]<self.goal[1]+self.goal_boundary and self.agent[1]>self.goal[1]-self.goal_boundary and self.agent[2]<self.goal[2]+self.goal_boundary and self.agent[2]>self.goal[2]-self.goal_boundary:
            print("Check: Reach the goal")
            log_f.write('Check: goal\n')
            log_f.flush()
            reward = 100

            # derive the power and energy, 1 Hz
            # Power model contains [height, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, wind_dir, wind_vel]
            # I will update the model that contains payloads.
            p = [self.agent[2], 0, 0, 0, 0 - action[0], 0 - action[1], 0 - action[2], self.wind_dir, self.wind_vel]
            p = self.norm(p, self.d_max, self.d_min)
            p = torch.tensor(p, dtype=torch.float32, device=device)
            p = model(p)

            self.energy += float(p)

            done = True
            success = True
            
            return np.array(next_state, dtype=np.float64), state_3D, reward, done

        else:

            #Common process for each step

            # Calculate current power consumption of the drone
            p = [self.agent[2], action[0], action[1], action[2], action[0] - self.prev_action[0], action[1] - self.prev_action[1], action[2] - self.prev_action[2], self.wind_dir, self.wind_vel]
            p = self.norm(p, self.d_max, self.d_min)
            p = torch.tensor(p, dtype=torch.float32, device=device)
            p = model(p)

            # energy (1 Hz)
            self.energy += float(p)
            
            # goal - current position of the drone
            # if cal_distance (distance - next_distance) is positive, drone moves correct direction
            # Nor, we set the negative reward
            cal_distance = (self.distance - self.next_distance) / 15 # 15 m/s standardization
            
            # when initial sequence, we set simple rewards
            if cal_distance >= 0:
                #print(self.min_distance, self.next_distance)
                if self.min_distance > self.next_distance:
                    distance_reward = 2
                    self.min_distance = self.next_distance
                else:
                    distance_reward = 1
            else:
                distance_reward = cal_distance
            
            pw_constant = 0.022
            power_reward = math.exp(pw_constant * (self.energy/10000)) * -1
            # When summation of energy is 20kJ, maximum reward is 101.
            #power_reward = -1 * float(p) / 1000 # 1 kJ standardization

            height_term = 0

            if self.agent[2] > self.goal[2]:
                height_term = (self.agent[2] - self.goal[2]) / 100

            # Each step reward

            #print("distance_reward: {}".format(distance_reward))
            #print("power_reward: {}".format(power_reward))
            print("energy: {}".format(self.energy/10000))
            #print("height_reward: {}".format(-1*height_term))
            print("to_goal: {}".format(self.distance))
            print("x:{}, y:{}, z:{}".format(action[0], action[1], action[2]))
            # It will be changed by (guide reward + power reward) 
            reward = distance_reward + power_reward - height_term

            if step == max_step-1:
                print("Check: Can't reach during maximum steps")

            done = False
            success = False


        '''
        Checksum termianl conditions
        Drones overcome the map, collide with obstcles, and approach the goal.

        if self.agent[0]<0 or self.agent[0]>=400 or self.agent[1]<0 or self.agent[1]>=400 or self.agent[2]<0 or self.agent[2]>=60:
            print("Check: Out of scope")
            reward = -100
            done = True
            success = False
            
            return np.array(next_state, dtype=np.float64), state_3D, reward, done

        elif step == max_step-1:
            print("Check: Can't reach during maximum steps")
            #reward = 50 * (1 - self.distance/self.toward_distance) - self.energy / (step* 1000)
            reward = -10
            done = True
            success = False

            return np.array(next_state, dtype=np.float64), state_3D, reward, done

        elif self.env_matrix[agent_int[0]][agent_int[1]][agent_int[2]] == 1:
            print("Check: Collide with obstacles")
            reward = -100
            done = True
            success = False

            return np.array(next_state, dtype=np.float64), state_3D, reward, done

        elif self.agent[0]<self.goal[0]+self.goal_boundary and self.agent[0]>self.goal[0]-self.goal_boundary and self.agent[1]<self.goal[1]+self.goal_boundary and self.agent[1]>self.goal[1]-self.goal_boundary and self.agent[2]<self.goal[2]+self.goal_boundary and self.agent[2]>self.goal[2]-self.goal_boundary:
            print("Check: Reach the goal")
            log_f.write('Check: goal\n')
            log_f.flush()
            reward = 100

            # derive the power and energy, 1 Hz
            # Power model contains [height, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, wind_dir, wind_vel]
            # I will update the model that contains payloads.
            p = [self.agent[2], 0, 0, 0, 0 - action[0], 0 - action[1], 0 - action[2], self.wind_dir, self.wind_vel]
            p = self.norm(p, self.d_max, self.d_min)
            p = torch.tensor(p, dtype=torch.float32, device=device)
            p = model(p)

            self.energy += float(p)

            done = True
            success = True
            
            return np.array(next_state, dtype=np.float64), state_3D, reward, done

        else:

            #Common process for each step

            p = [self.agent[2], action[0], action[1], action[2], action[0] - self.prev_action[0], action[1] - self.prev_action[1], action[2] - self.prev_action[2], self.wind_dir, self.wind_vel]
            p = self.norm(p, self.d_max, self.d_min)
            p = torch.tensor(p, dtype=torch.float32, device=device)
            p = model(p)
            
            distance_reward = (self.distance - self.next_distance) / 15 # 15 m/s standardization
            power_reward = -1 * float(p) / 1000 # 1 kJ standardization

            # standardization through mean of velocity and power consumption.

            self.beta = step/max_step
            if self.beta < 0.4:
                self.beta = 0.4
            height_term = 0

            if self.agent[2] > self.goal[2]:
                height_term = (self.agent[2] - self.goal[2]) / 10 

            # energy (1 Hz)
            self.energy += float(p)
            # It will be changed by (guide reward + power reward) 
            reward = distance_reward * (1 - self.beta) + power_reward * self.beta - height_term

            done = False
            success = False
        '''

        self.scatter_array.append(ax.scatter(self.agent[0], self.agent[1], self.agent[2], c='coral', s=15))

        #if done:
        #    print('total energy = %f' % self.energy)
        #    log_f.write('{}\n'.format(self.energy))
        #    log_f.flush()

        '''
        Draw function
        It uses scatter graph currently.
        I will change line graph instead of scatter.

        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.01)
        '''

        # save the action as a previous action for calculating acceleration.
        self.prev_action = action
        self.distance = self.next_distance
        # step count
        self.step_cnt += 1

        '''
        All step returns drone's next state (action, acc, etc.), arround information, reward, and terminal information.
        '''
        
        return np.array(next_state, dtype=np.float64), state_3D, reward, done


    def reset(self, episode, wind_matrix):

        # Clean the scatter graph
        for elem in self.scatter_array:
            elem.remove()

        self.energy = 0
        self.env_matrix = np.zeros((400,400,60))
        self.agent = np.array([0, 0, 0])
        self.step_cnt = 0
        self.prev_action = [0.0, 0.0, 0.0]
        self.goal_gen_boundary = [400, 400, 60]

        # Obstacles reconstuction
        for i in range(len(N)):
            for z in range(D[i], dD[i]):
                for x in range(N[i], N[i]+dN[i]):
                    for y in range(E[i], E[i]+dE[i]):
                        self.env_matrix[x][y][z] = 1

        # Agent generation
        while True:
            self.agent = np.array([random.randint(0, 400-1), random.randint(0, 400-1), random.randint(0, 60-1)])
            if self.env_matrix[self.agent[0]][self.agent[1]][self.agent[2]] != 1:
                break

        # Goal boundary max(0) ~ min(400)
        goal_low = [max(0, self.agent[0] - self.goal_gen_boundary[0]), max(0, self.agent[1] - self.goal_gen_boundary[1]), max(0, self.agent[2] - self.goal_gen_boundary[2])]
        goal_high = [min(400, self.agent[0] + self.goal_gen_boundary[0]), min(400, self.agent[1] + self.goal_gen_boundary[1]), min(60, self.agent[2] + self.goal_gen_boundary[2])]

        while True:
            self.goal = np.array([random.randint(goal_low[0], goal_high[0]-1), random.randint(goal_low[1], goal_high[1]-1), random.randint(goal_low[2], goal_high[2]-1)])
            if self.env_matrix[self.agent[0]][self.agent[1]][self.agent[2]] != 1:
                break

        '''
        It can determine agent and goal generation manually.
        for example, 
        self.agent = np.array([180, 380, 10], dtype=np.float64) 
        self.goal = np.array([340, 40, 10], dtype=np.float64)
        '''

        self.scatter_array = []
        self.scatter_array.append(ax.scatter(self.agent[0], self.agent[1], self.agent[2], c='red', s=40))
        self.scatter_array.append(ax.scatter(self.goal[0], self.goal[1], self.goal[2], c='green', s=40))

        agent_int = self.agent.astype(int)
        agent_for_wind = np.clip(agent_int, [0, 0, 0], [400-1, 400-1, 60-1])
            
        self.wind_dir = wind_matrix[int(agent_for_wind[0]/20)][1][int(agent_for_wind[2]/2)][int(agent_for_wind[1]/20)]
        self.wind_vel = wind_matrix[int(agent_for_wind[0]/20)][0][int(agent_for_wind[2]/2)][int(agent_for_wind[1]/20)]
            
        self.distance = math.sqrt(((self.goal[0]-self.agent[0])**2) + ((self.goal[1]-self.agent[1])**2) + ((self.goal[2]-self.agent[2])**2))
        self.min_distance = self.distance
        self.next_distance = 0
        self.toward_distance = self.distance

        new_state = np.array([self.agent[0]/400, self.agent[1]/400, self.agent[2]/60, 0.0, 0.0, 0.0, self.goal[0]/400, self.goal[1]/400, self.goal[2]/60, self.wind_dir/360, self.wind_vel/10], dtype=np.float64)
        conv_3D = self.get_3D_state()

        '''
        Step function returns initial state of the drone and its observation for 3D convolution.
        '''

        return new_state, conv_3D