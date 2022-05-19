import random
import numpy as np

from collections import deque

class ReplayBuffer:
    #def __init__(self, batch_size, max_size, device):
    # for TD3 with BC    
    def __init__(self, batch_size, max_size, state1_dim, state2_dim, action_dim, device):
        self.max_size = int(max_size)
        self.buffer = deque(maxlen = self.max_size)
        self.batch_size = batch_size
        self.size = 0

        # For TD3 with BC
        '''
        self.state1 = np.zeros((max_size, state1_dim))
        self.state2 = np.zeros((max_size, state2_dim))
        self.acton = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state1 = np.zeros((max_size, state1_dim))
        self.next_state2 = np.zeros((max_size, state2_dim))
        self.done = np.zeros((max_size, 1))
        '''

        self.state1 = 0
        self.state2 = 0
        self.acton = 0
        self.reward = 0
        self.next_state1 = 0
        self.next_state2 = 0
        self.done = 0
        
    '''
    def add(self, transition):
        # transition is tuple of (state1, state2, action, reward, next_state1, next_state2, done)
        self.buffer.append(transition)
        self.size = min(self.size+1, self.max_size)
    '''

    def sample(self):

        '''
        mini_batch = random.sample(self.buffer, self.batch_size)
        mini_batch = np.array(mini_batch, dtype= object)

        state1 = np.vstack(mini_batch[:,0])
        state2 = np.vstack(mini_batch[:,1])
        action = list(mini_batch[:,2])
        reward = list(mini_batch[:,3])
        next_state1 = np.vstack(mini_batch[:,4])
        next_state2 = np.vstack(mini_batch[:,5])
        done = list(mini_batch[:,6])
        '''
        index = np.random.randint(0, self.size, self.batch_size)

        return dict(
            state1 = np.array(self.state1[index]),
            state2 = np.array(self.state2[index]),
            action = np.array(self.action[index]), 
            reward = np.array(self.reward[index]), 
            next_state1 = np.array(self.next_state1[index]),
            next_state2 = np.array(self.next_state2[index]),
            done = np.array(self.done[index]), 
        )

    def convert_ARRAY(self, dataset):

        dataset = np.array(dataset, dtype=object)

        self.state1 = np.vstack(dataset[:,0])
        self.state2 = np.vstack(dataset[:,1])
        self.action = np.vstack(dataset[:,2])
        self.reward = np.vstack(dataset[:,3])
        self.next_state1 = np.vstack(dataset[:,4])
        self.next_state2 = np.vstack(dataset[:,5])
        self.done = np.vstack(dataset[:,6])
        
        self.size = self.state1.shape[0]


    def normalize_states(self, eps = 1e-3):
        mean1 = self.state1.mean(0, keepdims = True)
        std1 = self.state1.std(0, keepdims = True) + eps
        mean2 = self.state2.mean(0, keepdims = True)
        std2 = self.state2.std(0, keepdims = True) + eps
        self.state1 = (self.state1 - mean1) / std1
        self.state2 = (self.state2 - mean2) / std2
        self.next_state1 = (self.next_state1 - mean1) / std1
        self.next_state2 = (self.next_state2 - mean2) / std2
        
        return mean1, std1, mean2, std2

    def __len__(self):
        return self.size