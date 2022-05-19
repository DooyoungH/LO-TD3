import random
import numpy as np

from collections import deque

class ReplayBuffer:
    def __init__(self, batch_size, max_size, device):
        self.max_size = int(max_size)
        self.buffer = deque(maxlen = self.max_size)
        self.batch_size = batch_size
        self.size = 0

        '''
        # For TD3 with BC
        self.state1 = np.zeros((max_size, state1_dim))
        self.state2 = np.zeros((max_size, state2_dim))
        self.acton = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state1 = np.zeros((max_size, state1_dim))
        self.next_state2 = np.zeros((max_size, state2_dim))
        self.done = np.zeros((max_size, 1))
        '''
        

    def add(self, transition):
        # transition is tuple of (state1, state2, action, reward, next_state1, next_state2, done)
        self.buffer.append(transition)
        self.size = min(self.size+1, self.max_size)

    def sample(self):

        mini_batch = random.sample(self.buffer, self.batch_size)
        mini_batch = np.array(mini_batch, dtype= object)

        state1 = np.vstack(mini_batch[:,0])
        state2 = np.vstack(mini_batch[:,1])
        action = list(mini_batch[:,2])
        reward = list(mini_batch[:,3])
        next_state1 = np.vstack(mini_batch[:,4])
        next_state2 = np.vstack(mini_batch[:,5])
        done = list(mini_batch[:,6])

        return dict(
            state1 = np.array(state1),
            state2 = np.array(state2),
            action = np.array(action), 
            reward = np.array(reward), 
            next_state1 = np.array(next_state1),
            next_state2 = np.array(next_state2),
            done = np.array(done), 
        )

    def __len__(self):
        return self.size