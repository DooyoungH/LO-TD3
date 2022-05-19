import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Normal

import numpy as np
import copy


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.conv_layer2 = self._conv_layer_set(16,32)

        self.fc1 = nn.Linear(state_dim -1 + 864, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_acition = max_action

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size = (3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer

    def forward(self, drone_stat, state_3D):
        b = self.conv_layer1(state_3D)
        b = self.conv_layer2(b)
        b = b.view(b.size(0), -1)
        a = F.relu(self.fc1(torch.cat([drone_stat, b], 1)))
        a = F.relu(self.fc2(a))
        
        action = torch.tanh(self.fc3(a))

        return action #* self.max_action


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.conv_layer2 = self._conv_layer_set(16,32)

         # Q1 architecture
        self.fc1 = nn.Linear(state_dim - 1 + action_dim + 864, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim - 1 + action_dim + 864, 400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)


    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size = (3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer


    def forward(self, drone_stat, state_3D, action):
        b = self.conv_layer1(state_3D)
        b = self.conv_layer2(b)
        b = b.view(b.size(0), -1)

        state_action = torch.cat([drone_stat, b, action], 1)

        Q1 = F.relu(self.fc1(state_action))
        Q1 = F.relu(self.fc2(Q1))
        Q1_value = self.fc3(Q1)

        Q2 = F.relu(self.fc4(state_action))
        Q2 = F.relu(self.fc5(Q2))
        Q2_value = self.fc6(Q2)

        return Q1_value, Q2_value


    def Q1(self, drone_stat, state_3D, action):
        b = self.conv_layer1(state_3D)
        b = self.conv_layer2(b)
        b = b.view(b.size(0), -1)

        state_action = torch.cat([drone_stat, b, action], 1)

        Q1 = F.relu(self.fc1(state_action))
        Q1 = F.relu(self.fc2(Q1))
        Q1_value = self.fc3(Q1)

        return Q1_value

class TD3_BC:

    def __init__(self, AT_LR, CR_LR, state_dim, action_dim, max_action, gamma, tau, policy_noise, noise_clip, policy_freq, alpha, device):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=AT_LR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CR_LR)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.steps = 0

        # new term
        self.total_it = 0

    def select_action(self, state1, state2, device):

        state1 = torch.FloatTensor(state1.reshape(1,-1)).to(device)
        state2 = torch.FloatTensor(state2.reshape(1,1,20,20,20)).to(device)

        return self.actor(state1, state2).cpu().data.numpy().flatten()

    def update(self, buffer, batch_size, device):
        self.total_it += 1
        self.steps += 1

        # Sample replay buffer
        samples = buffer.sample()

        state1 = torch.FloatTensor(samples["state1"].reshape(batch_size,-1)).to(device)
        state2 = torch.FloatTensor(samples["state2"].reshape(batch_size,1,20,20,20)).to(device)
        actions = torch.FloatTensor(samples["action"]).to(device)
        reward = torch.FloatTensor(samples["reward"]).reshape((batch_size,1)).to(device)
        next_state1 = torch.FloatTensor(samples["next_state1"].reshape(batch_size,-1)).to(device)
        next_state2 = torch.FloatTensor(samples["next_state2"].reshape(batch_size,1,20,20,20)).to(device)
        done = torch.FloatTensor(samples["done"]).reshape((batch_size,1)).to(device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # noise => not confirmed
            next_actions = (self.actor_target(next_state1, next_state2) + noise).clamp(-self.max_action, self.max_action)
            # next_actions => not confirmed

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state1, next_state2, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q) #.detach() ?

        # Get current Q value estimation
        current_Q1, current_Q2 = self.critic(state1, state2, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy update
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state1, state2)
            Q = self.critic.Q1(state1, state2, pi)
            lmbda = self.alpha/Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, actions)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update of target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1- self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1- self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

            




