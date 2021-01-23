import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, normal, MultivariateNormal
import gym
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def remember(self, state, action, logprob, reward, is_terminal):
        state = torch.from_numpy(state).float().to(device)
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprob)
        # if reward==0:
        #     reward = 50
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, checkpoint):
#         super(Actor, self).__init__()

#         self.action_dim = action_dim
#         self.state_dim = state_dim
#         self.checkpoint = checkpoint+'/actor_ddpg'
        
#         self.fc1 = nn.Linear(state_dim, 32)
#         self.fc2 = nn.Linear(32,64)
#         self.fc3 = nn.Linear(64,32)
#         self.mu = nn.Linear(32,action_dim)

#         self.bn1 = nn.LayerNorm(32)
#         self.bn2 = nn.LayerNorm(64)
#         self.bn3 = nn.LayerNorm(32)

#         fc1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])
#         fc2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])
#         fc3 = 1.0/np.sqrt(self.fc3.weight.data.size()[0])
#         mu = 0.03

#         torch.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc3.weight.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.fc3.bias.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.mu.weight.data, -mu, mu)
#         torch.nn.init.uniform_(self.mu.bias.data, -mu, mu)


#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = torch.tanh(self.mu(x))

#         return x.to(device)
    
#     def save(self):
#         torch.save(self.state_dict(), self.checkpoint)

#     def load(self):
#         self.load_state_dict(torch.load(self.checkpoint))


# class Critic(nn.Module):
#     def __init__(self, state_dim, checkpoint):
#         super(Critic, self).__init__()

#         self.state_dim = state_dim
#         self.checkpoint = checkpoint+'/critic_ddpg'

#         self.fc1 = nn.Linear(state_dim, 64)
#         self.bn1 = nn.LayerNorm(64)
#         self.fc2 = nn.Linear(64,128)
#         self.bn2 = nn.LayerNorm(128)
#         self.fc3 = nn.Linear(128,32)
#         self.bn3 = nn.LayerNorm(32)
#         self.pi = nn.Linear(32,1)

#         fc1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])
#         fc2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])
#         fc3 = 1.0/np.sqrt(self.fc3.weight.data.size()[0])
#         pi = 0.03

#         torch.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)
#         torch.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)
#         torch.nn.init.uniform_(self.fc3.weight.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.fc3.bias.data, -fc3, fc3)
#         torch.nn.init.uniform_(self.pi.weight.data, -pi, pi)
#         torch.nn.init.uniform_(self.pi.bias.data, -pi, pi)


#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = self.pi(x)

#         return x.to(device)

#     def save(self):
#         torch.save(self.state_dict(), self.checkpoint)

#     def load(self):
#         self.load_state_dict(torch.load(self.checkpoint))


# class ActorCritic:
#     def __init__(self, state_dim, action_dim, n_latent_var, checkpoint='tmp/ppo', beta=0.001):
#         # super(ActorCritic, self).__init__()
#         self.action_dim = action_dim
#         self.std = nn.Parameter(torch.zeros(1,self.action_dim)).to(device)
#         # actor
#         self.action_layer = Actor(state_dim,action_dim, checkpoint).to(device)
#         self.value_layer = Critic(state_dim, checkpoint).to(device)
#         self.old_actor = Actor(state_dim,action_dim, checkpoint).to(device)
#         self.old_actor.load_state_dict(self.action_layer.state_dict())

#         self.actor_optimizer = torch.optim.Adam(self.action_layer.parameters(), lr=beta)
#         self.critic_optimizer = torch.optim.Adam(self.value_layer.parameters(), lr=beta)

#         # self.device = torch.device('cuda' if T.cuda.is_available() else 'cpu')
#         # critic
#         # self.value_layer = nn.Sequential(
#         #         nn.Linear(state_dim, n_latent_var),
#         #         nn.Tanh(),
#         #         nn.Linear(n_latent_var, n_latent_var),
#         #         nn.Tanh(),
#         #         nn.Linear(n_latent_var, 1)
#         #         )
#         self.checkpoint = checkpoint
        
#     # def forward(self):
#     #     raise NotImplementedError
        
#     def act(self, state):
#         with torch.no_grad():
#             state = torch.from_numpy(state).float().to(device)
#             # with torch.no_grad(): 
#             action_dist = self.old_actor.forward(state)
#             # print(action_probs)
#             # print('action_dist: ',action_dist)

#             dist = normal.Normal(action_dist, self.std.exp())
#             # print(dist.device())
#             action = dist.sample()[0]
#             log_prob = dist.log_prob(action)[0]
#         # memory.states.append(state)
#         # memory.actions.append(action)
#         # memory.logprobs.append(dist.log_prob(action)[0])
#         return action, log_prob
    
#     def evaluate(self, state, action):
#         action_dist = self.action_layer.forward(state)
#         dist = normal.Normal(action_dist, self.std.exp())
#         # action = dist.sample()[0]
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
        
#         state_value = self.value_layer.forward(state)
        
#         return action_logprobs, torch.squeeze(state_value), dist_entropy

#     def save(self):
#         print('saving into {}'.format(self.checkpoint))
#         if not os.path.isdir(self.checkpoint):
#             os.makedirs(self.checkpoint)
#         self.action_layer.save()
#         self.value_layer.save()
#         # torch.save(self.action_layer.state_dict(), self.checkpoint+'/action.pt')
#         # torch.save(self.value_layer.state_dict(), self.checkpoint+'/value.pt')

#     def load(self):
#         print('loading from {}'.format(self.checkpoint))
#         if not os.path.isdir(self.checkpoint):
#             return
#         self.action_layer.load()
#         self.value_layer.load()
#         # self.action_layer.load_state_dict(torch.load(self.checkpoint+'/action.pt'))
#         # self.value_layer.load_state_dict(torch.load(self.checkpoint+'/value.pt'))

#     def optimize(self, actor_loss, critic_loss):
#         self.action_layer.train()
#         self.value_layer.train()

#         self.actor_optimizer.zero_grad()
#         self.critic_optimizer.zero_grad()
#         # print(critic_loss)
#         actor_loss.mean().backward(retain_graph=True)
#         critic_loss.backward(retain_graph=True)

#         self.actor_optimizer.step()
#         self.critic_optimizer.step()

#     def transfer(self):
#         self.old_actor.load_state_dict(self.action_layer.state_dict())

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, checkpoint='tmp/ppo'):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                # nn.Softmax(dim=-1)
                nn.Tanh()
                )
        self.std = nn.Parameter(torch.zeros(1,action_dim)).to(device)
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        self.checkpoint = checkpoint
        self.action_var = torch.full((action_dim,), 0.5*0.5).to(device)
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        cov_mat = torch.diag(self.action_var).to(device)
        # print(action_probs)
        dist = MultivariateNormal(action_probs, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print('action',action)
        # memory.states.append(state)
        # memory.actions.append(action)
        # memory.logprobs.append(log_prob)
        return action, log_prob
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        action_var = self.action_var.expand_as(action_probs)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_probs, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self):
        print('saving into {}'.format(self.checkpoint))
        if not os.path.isdir(self.checkpoint):
            os.makedirs(self.checkpoint)
        torch.save(self.action_layer.state_dict(), self.checkpoint+'/action.pt')
        torch.save(self.value_layer.state_dict(), self.checkpoint+'/value.pt')

    def load(self):
        print('loading from {}'.format(self.checkpoint))
        if not os.path.isdir(self.checkpoint):
            return
        self.action_layer.load_state_dict(torch.load(self.checkpoint+'/action.pt'))
        self.value_layer.load_state_dict(torch.load(self.checkpoint+'/value.pt'))
        

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, checkpoint='tmp/ppo'):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.checkpoint = checkpoint
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, checkpoint+'/policy').to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, checkpoint+'/old_policy').to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state)
    
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        print('update')
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:a
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            # print(logprobs)
            # print(old_logprobs)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:

            advantages = rewards - state_values.detach()

            critic_loss = self.MseLoss(state_values, rewards)
            # print(critic_loss)
            surr1 = ratios * advantages 
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*critic_loss - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        print('loss: ', loss.mean())    
        # take gradient step
            # self.policy.optimize(loss, critic_loss)
        # self.old_actor.load_state_dict(self.action_layer.state_dict())
        # self.policy.transfer()


    def save(self):
        print('saving to checkpoint................................')
        self.policy.save()
        # self.policy_old.save()

    def load(self):
        print('loading data........................................')
        self.policy.load()
        # self.policy_old.load()