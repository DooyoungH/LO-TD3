import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Normal

import numpy as np


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, log_std_min = -20, log_std_max = 2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.conv_layer2 = self._conv_layer_set(16,32)

        self.fc1 = nn.Linear(state_dim -1 + 864, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.fc4 = nn.Linear(300, action_dim)

        self.max_action = max_action

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

        mu = self.fc3(a)
        log_std = self.fc4(a)
        # Action distribution clip
        log_std = torch.clamp(log_std, min=self.log_std_min, max= self.log_std_max)
        # Probability of policy Exp(Q)
        std = torch.exp(log_std)

        return mu, std


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.conv_layer2 = self._conv_layer_set(16,32)

        self.fc1 = nn.Linear(state_dim - 1 + action_dim + 864, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size = (3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer

    def forward (self, drone_stat, state_3D, action):
        b = self.conv_layer1(state_3D)
        b = self.conv_layer2(b)
        b = b.view(b.size(0), -1)

        state_action = torch.cat([drone_stat, b, action], 1)

        Q = F.relu(self.fc1(state_action))
        Q = F.relu(self.fc2(Q))
        Q_value = self.fc3(Q)

        return Q_value


class SAC:

    def __init__(self, AT_LR, CR_LR, state_dim, action_dim, max_action, gamma, target_entropy, log_alpha, alpha, tau, device):
        # max_action?
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=AT_LR)
        
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=CR_LR)
        
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=CR_LR)

        self.max_action = max_action
        self.action_bias = 0
        self.target_entropy = target_entropy
        self.log_alpha = log_alpha
        self.alpha = alpha
        self.alpha_optimizer = optim.Adam([log_alpha], lr=CR_LR)
        self.gamma = gamma
        self.tau = tau

        # For searching zero parameters of Actor model
        self.suma = 0
        self.ZERO_THRESHOLD = 5e-4


    def get_action(self, mu, std, action_scale, action_bias):
        normal = Normal(mu, std)
        # reparameterization trick (mean + std * N(0,1))
        z = normal.rsample()
        action = torch.tanh(z) * action_scale + action_bias

        return  action.data.cpu().numpy().flatten()


    def eval_action(self, mu, std, action_scale, action_bias, epsilon=1e-6):
        normal = Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)

        # Enforcing Action Bounds
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_policy = log_prob.sum(1, keepdim=True)

        return action * action_scale + action_bias, log_policy


    def update(self, buffer, batch_size, device):
        samples = buffer.sample()

        state1 = torch.FloatTensor(samples["state1"].reshape(batch_size,-1)).to(device)
        state2 = torch.FloatTensor(samples["state2"].reshape(batch_size,1,20,20,20)).to(device)
        actions = torch.FloatTensor(samples["action"]).to(device)
        reward = torch.FloatTensor(samples["reward"]).reshape((batch_size,1)).to(device)
        next_state1 = torch.FloatTensor(samples["next_state1"].reshape(batch_size,-1)).to(device)
        next_state2 = torch.FloatTensor(samples["next_state2"].reshape(batch_size,1,20,20,20)).to(device)
        done = torch.FloatTensor(samples["done"]).reshape((batch_size,1)).to(device)

        '''
        Critic update sequence
        '''
        criterion = torch.nn.MSELoss()

        # Get Q-values using two Q-functions to mitigate overestimation bias
        Q1 = self.critic_1(state1, state2, actions)
        Q2 = self.critic_2(state1, state2, actions)

        # Get target value
        mu, std = self.actor(next_state1, next_state2)
        next_actions, next_log_policy = self.eval_action(mu, std, self.max_action, self.action_bias)
        target_Q1 = self.critic_1_target(next_state1, next_state2, next_actions)
        target_Q2 = self.critic_2_target(next_state1, next_state2, next_actions)

        ########################## Debugging
        #target_Q1, target_Q2 => Nan
        # next_log_policy = Nan


        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = target_Q - self.alpha * next_log_policy
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # L1-term
        L1_reg = torch.tensor(0., requires_grad =True)
        for name, param in self.critic_1.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        Q1_loss = criterion(Q1, target_Q) + (1e-8 * L1_reg)
        self.critic_1_optimizer.zero_grad()
        Q1_loss.backward()
        self.critic_1_optimizer.step()

        # L1_term
        L1_reg = torch.tensor(0., requires_grad =True)
        for name, param in self.critic_2.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        Q2_loss = criterion(Q2, target_Q) + (1e-8 * L1_reg)
        self.critic_2_optimizer.zero_grad()
        Q2_loss.backward()
        self.critic_2_optimizer.step()

        '''
        Actor update sequence
        '''
        mu, std = self.actor(state1, state2)
        # log_policy = 1/exp(Q)
        action, log_policy = self.eval_action(mu, std, self.max_action, self.action_bias)
        
        Q1 = self.critic_1(state1, state2, action)
        Q2 = self.critic_2(state1, state2, action)

        min_Q = torch.min(Q1, Q2)

        L1_reg = torch.tensor(0., requires_grad =True)
        for name, param in self.actor.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        actor_loss = ((self.alpha * log_policy) - min_Q).mean() + (1e-8 * L1_reg)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #print(actor_loss)

        '''
        Alpha update Sequence
        '''

        alpha_loss = -(self.log_alpha * (log_policy + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = torch.exp(self.log_alpha)

        '''
        for p in self.actor.parameters():
            p = p.data.cpu().numpy()
            self.suma += (abs(p) < self.ZERO_THRESHOLD).sum()
        '''

        # Soft target update:
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_( (self.tau * target_param.data) + ((1-self.tau) * param.data))

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_( (self.tau * target_param.data) + ((1-self.tau) * param.data))


    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
                
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        print("model is loaded")