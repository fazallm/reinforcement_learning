import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class OUNoise(object):
    def __init__(self, mean, std_dev=0.15, theta=0.2, sigma=0.15, dt=1e-2, x0=None):
        self.mean = mean
        self.std_dev = std_dev
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0=x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)


class ReplayBuffer(object):
    def __init__(self, size, input_shape, n_actions):
        self.size = size
        self.counter = 0
        self.state = np.zeros((size, *input_shape))
        self.new_state = np.zeros((size, *input_shape))
        self.action = np.zeros((size, n_actions))
        self.reward = np.zeros(size)
        self.terminal_memory = np.zeros(self.size, dtype=np.float32)

    def store(self, state, new_state, action, reward, done):
        index = self.counter%self.size
        self.state[index] = state
        self.new_state[index] = new_state
        self.action[index] = action
        self.reward[index] = reward
        self.terminal_memory = 1-done
        self.counter += 1

    def sample(self):
        index = np.random(min(self.counter, self.size))
        return self.state[index], self.action[index], self.reward[index], self.new_state[index], self.terminal_memory[index]


class Critic(nn.Module):
    def __init__(self, input_shape, beta, n_actions, name, checkpoint='tmp/ddpg'):
        super(Critic, self).__init__()
        self.name = name
        self.beta = beta
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lay1_dims = 400
        self.lay2_dims = 300
        self.checkpoint_file = os.path.join(checkpoint,name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_shape, self.lay1_dims)
        self.fc2 = nn.Linear(self.lay1_dims, self.lay2_dims)
        self.fc3 = nn.Linear(self.n_actions, self.lay2_dims)
        self.q = nn.Linear(self.lay2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = F.re;u(state_value)
        state_value = self.fc2(state_value)
        action_value = F.relu(self.fc3(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save(self):
        T.save(self.state_dict(), self.checkpoint)

    def load(self):
        self.load_state_dict(T.load(self.checkpoint))

class Actor(nn.Module):
    def __init__(self, input_shape, n_actions, alpha, name, checkpoint='tmp/ddpg'):
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.alpha = alpha
        self.n_actions = n_actions
        self.checkpoint = os.path.join(checkpoint,name+'_ddpg')

        self.fc1 = nn.Linear(*input_shape, 400)
        self.fc2 = nn.Linear(400,300)
        self.policy = nn.Linear(300, self.n_actions)
        self.optimizer = optim.Actor(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x - F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action = T.tanh(self.policy(x))

        return action

    def save(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint))


class Agent(object):
    def __init__(self, alpha, beta, input_shape, tau, env, gamma=0.99, n_actions=2, size=100000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(input_shape, n_actions, alpha, 'Actor')
        self.target_actor = Actor(input_shape, n_actions, alpha, 'TargetActor')

        self.critic = Critic(input_shape, beta, n_actions, 'Critic')
        self.target_critic = Critic(input_shape, beta, n_actions, 'TargetCritic')

        self.noise = OUNoise(np.zeros(n_actions))

        self.memory = ReplayBuffer(size, input_shape, n_actions)


    def update(self, tau):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        done = T.tensor(done, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target=[]
        for i in range(self.batch_size):
            target.append(reward[i]+self.gamma * critic_value_[i]*done[i])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        policy = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, policy)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update()

    def remember(self, state, action, reward, newstate, done):
        self.memory.store(state, new_state, action, reward, done)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()

