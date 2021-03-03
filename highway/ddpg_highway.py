import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import math
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from gym.spaces import Box,Discrete
# import tensorflow as tf
import space


# def _reward(self, action: Action) -> float:
#         """
#         The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
#         :param action: the last action performed
#         :return: the corresponding reward
#         """
#         neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
#         lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
#             else self.vehicle.lane_index[2]
#         scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
#         reward = \
#             + self.config["collision_reward"] * self.vehicle.crashed \
#             + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
#             + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
#         reward = utils.lmap(reward,
#                           [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
#                           [0, 1])
#         reward = 0 if not self.vehicle.on_road else reward
#         return reward


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

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mean, self.sigma)

class ReplayBuffer(object):
    def __init__(self, size, input_shape, n_actions):
        self.size = size
        self.counter = 0
        self.state = np.zeros((size, *input_shape))
        self.new_state = np.zeros((size, *input_shape))
        self.action = np.zeros((size,1))
        self.reward = np.zeros(size)
        self.terminal_memory = np.zeros(self.size, dtype=np.float32)

    def store(self, state, new_state, action, reward, done):
        index = self.counter%self.size
        self.state[index] = state
        self.new_state[index] = new_state
        self.action[index] = [action]
        self.reward[index] = reward
        self.terminal_memory[index] = 1-done
        self.counter += 1

    def sample(self, batch_size):
        index = np.random.choice(min(self.counter, self.size), batch_size)

        state = self.state[index]
        action = self.action[index]
        reward = self.reward[index]
        new_state = self.new_state[index]
        # print(self.terminal_memory)
        terminal_memory = self.terminal_memory[index]
        return state,action, reward, new_state, terminal_memory

class Critic(nn.Module):
    def __init__(self, input_shape, beta, n_actions, name, checkpoint='tmp/ddpg'):
        super(Critic, self).__init__()
        self.name = name
        self.beta = beta
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lay1_dims = 32
        self.lay2_dims = 64
        self.checkpoint = os.path.join(checkpoint,name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_shape, self.lay1_dims)
        self.bn1 = nn.LayerNorm(self.lay1_dims)
        fc1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])

        self.fc2 = nn.Linear(self.lay1_dims, self.lay2_dims)
        self.bn2 = nn.LayerNorm(self.lay2_dims)
        fc2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])

        self.fc3 = nn.Linear(1, self.lay2_dims)
        fc3 = 0.003

        self.q = nn.Linear(self.lay2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        T.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
        T.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)
        T.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
        T.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)
        T.nn.init.uniform_(self.q.weight.data, -fc3, fc3)
        T.nn.init.uniform_(self.q.bias.data, -fc3, fc3)



        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        # print('action',action.shape)
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

        self.fc1 = nn.Linear(*input_shape, 32)
        self.bn1 = nn.LayerNorm(32)
        fc1 = 1.0/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc2 = nn.Linear(32,64)
        self.bn2 = nn.LayerNorm(64)
        fc2 = 1.0/np.sqrt(self.fc2.weight.data.size()[0])
        self.policy = nn.Linear(64, n_actions)
        fc3 = 0.003
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        T.nn.init.uniform_(self.fc1.weight.data, -fc1, fc1)
        T.nn.init.uniform_(self.fc1.bias.data, -fc1, fc1)
        T.nn.init.uniform_(self.fc2.weight.data, -fc2, fc2)
        T.nn.init.uniform_(self.fc2.bias.data, -fc2, fc2)
        T.nn.init.uniform_(self.policy.weight.data, -fc3, fc3)
        T.nn.init.uniform_(self.policy.bias.data, -fc3, fc3)

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x - F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        action = T.sigmoid(self.policy(x))

        return action

    def save(self):
        T.save(self.state_dict(), self.checkpoint)

    def load(self):
        self.load_state_dict(T.load(self.checkpoint))


# class Inverter:
#     def __init__(self, action_bounds):

#         self.sess = tf.compat.v1.Session()       
        
#         self.action_size = len(action_bounds[0])
        
#         self.action_input = tf.placeholder(tf.float32, [None, self.action_size])
#         self.pmax = tf.constant(action_bounds[0], dtype = tf.float32)
#         self.pmin = tf.constant(action_bounds[1], dtype = tf.float32)
#         self.prange = tf.constant([x - y for x, y in zip(action_bounds[0],action_bounds[1])], dtype = tf.float32)
#         self.pdiff_max = tf.div(-self.action_input+self.pmax, self.prange)
#         self.pdiff_min = tf.div(self.action_input - self.pmin, self.prange)
#         self.zeros_act_grad_filter = tf.zeros([self.action_size])
#         self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])
#         self.grad_inverter = tf.where(tf.greater(self.act_grad, self.zeros_act_grad_filter), tf.multiply(self.act_grad, self.pdiff_max), tf.multiply(self.act_grad, self.pdiff_min))        
    
#     def invert(self, grad, action):    
#         return self.sess.run(self.grad_inverter, feed_dict = {self.action_input: action, self.act_grad: grad[0]})


class Agent(object):
    def __init__(self, alpha, beta, tau, env, gamma=0.99, size=1000000, batch_size=64, checkpoint='tmp/ddpg'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.env = env
        self.input_shape = env.reset().reshape(-1).shape[0]
        self.checkpoint = checkpoint
        self.continious_action_space=False

        if isinstance(env.action_space, Box):
            self.n_actions = env.action_space.shape[0]
            self.high = env.action_space.high
            self.low = env.action_space.low
        else:
            self.n_actions = 1
            self.high = [env.action_space.n]
            self.low = [0]
        # print(self.n_actions)
        # self.high = self.env.action_space.high
        # self.inverter = Inverter([self.high, self.low])
        self.actor = Actor([self.input_shape], self.n_actions, alpha, 'Actor', checkpoint = checkpoint)
        self.target_actor = Actor([self.input_shape], self.n_actions, alpha, 'TargetActor',checkpoint = checkpoint)

        self.critic = Critic([self.input_shape], beta, self.n_actions, 'Critic', checkpoint=checkpoint)
        self.target_critic = Critic([self.input_shape], beta, self.n_actions, 'TargetCritic', checkpoint=checkpoint)

        self.noise = OUNoise(np.zeros(1))

        self.memory = ReplayBuffer(size, [self.input_shape], self.n_actions)
        self.update(tau=1)


    def update(self, tau=None):
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
        # print('action',action)
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
        self.memory.store(state, newstate, action, reward, done)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        # print(mu)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        action = np.clip(mu_prime.cpu().detach().numpy(), 0, 1)
        self.actor.train()
        return action

    def save_models(self, scores):
        if not os.path.isdir(self.checkpoint):
            os.makedirs(self.checkpoint)
        self.actor.save()
        self.target_actor.save()
        self.critic.save()
        self.target_critic.save()
        np.savetxt(os.path.abspath(self.checkpoint)+'/data.csv', scores, delimiter=',')

    def load_models(self):
        self.actor.load()
        self.target_actor.load()
        self.critic.load()
        self.target_critic.load()

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

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
        lane = self.env.vehicle.target_lane_index[2] if isinstance(self.env.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
            + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                        [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                        [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward


    def runner(self, i_episode, render=False, train=True):
    #     env = gym.make(env_name)
    # # print(env.actions())
    # # pass
    

    # # agent.load_models()
    #     np.random.seed(0)

        if not train:
            self.load_models()
        score_history = []
        scores=[]
        # self.load_models()
        # scores = np.genfromtxt(self.checkpoint+'/data.csv', delimiter=',')
        for i in range(1,i_episode+1):
            obs = self.env.reset()
            done = False
            score = 0
            step = 0
            while not done:
                step+=1
                # print(obs)
                obs = obs.reshape(-1)
                act = self.choose_action(obs)
                # print(act)
                # print(action)
                new_state, reward, done, info = self.env.step(act.detach().cpu().numpy()[0])
                if train:
                    self.remember(obs, act, reward, new_state.reshape(-1), int(done))
                    self.learn()
                score += reward
                obs = new_state
                if render:
                    self.env.render()
                # if done:
                #     break
            scores.append(score)

            if (i % 25 == 0) and (train==True):
                self.save_models(scores)

            print('episode ', i, 'score %.2f' % score,
                'trailing 50 games avg %.3f' % np.mean(scores[-50:]),
                'finished after ', step, ' episode')
        self.env.close()
        
        # self.save_models(scores)
        return scores

class WolpertingerAgent(Agent):

    def __init__(self, alpha, beta, tau, env, gamma=0.99, size=1000000, batch_size=64, checkpoint='tmp/ddpg', max_actions=1e6, k_ratio=0.1):
        super().__init__(alpha, beta, tau, env, gamma, size, batch_size, checkpoint)
        self.experiment = env.spec.id
        if self.continious_action_space:
            self.action_space = space.Space(self.low, self.high, max_actions)
            max_actions = self.action_space.get_number_of_actions()
        else:
            max_actions = int(env.action_space.n)*5
            self.action_space = space.Discrete_space(max_actions)

        self.k_nearest_neighbors = max(5, int(max_actions * k_ratio))
        # print(self.k_nearest_neighbors)
    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def choose_action(self, state):
        # taking a continuous action from the actor
        proto_action = super().choose_action(state)
        # if self.k_nearest_neighbors < 1:
        #     return proto_action
        # print('proto ',proto_action)
        # return the best neighbor of the proto action
        state = T.tensor(state, dtype=T.float32)
        return self.wolp_action(state, proto_action)

    def wolp_action(self, state, proto_action):
        # get the proto_action's k nearest neighbors
        raw_action,actions = self.action_space.search_point(proto_action, self.k_nearest_neighbors)
        # actions=actions[0]
        # self.data_fetch.set_ndn_action(actions[0].tolist())
        # make all the state, action pairs for the critic
        states = T.tensor(np.tile(state, [len(raw_action), 1]))
        # evaluate each pair through the critic
        raw_action = T.tensor(raw_action, dtype=T.float32)
        # print('action', actions)
        actions_evaluation = self.critic.forward(states, raw_action)
        # find the index of the pair with the maximum value
        max_index = np.argmax(actions_evaluation.detach().cpu().numpy())
        # return the best action
        return raw_action[max_index], actions[max_index]

    def runner(self, i_episode, render=False, train=True):
        if not train:
            self.load_models()
        score_history = []
        scores=[]
        # self.load_models()
        # scores = np.genfromtxt(self.checkpoint+'/data.csv', delimiter=',')
        for i in range(1,i_episode+1):
            obs = self.env.reset()
            done = False
            score = 0
            step = 0
            while not done:
                step+=1
                # print(obs)
                obs = obs.reshape(-1)
                raw, act = self.choose_action(obs)
                
                # print(action)
                new_state, reward, done, info = self.env.step(act[0]%5)
                new_reward = self._reward(act[0])
                if train:
                    self.remember(obs, raw, new_reward, new_state.reshape(-1), int(done))
                    # self.learn()
                score += reward
                obs = new_state
                if render:
                    self.env.render()
                # if done:
                #     break
            scores.append(score)
            self.learn()
            if (i % 25 == 0) and (train==True):
                self.save_models(scores)

            print('episode ', i, 'score %.2f' % score,
                'trailing 50 games avg %.3f' % np.mean(scores[-50:]),
                'finished after ', step, ' episode')
        self.env.close()
        
        # self.save_models(scores)
        return scores

    def learn(self):
        if self.memory.counter < self.batch_size:
            return
        # print('learn')
        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        done = T.tensor(done, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions=[]
        # target_actions = self.target_actor.forward(new_state)
        proto_action = self.target_actor.forward(new_state)
        # print('proto',proto_action)
        for i in range(len(proto_action)):
            x,y = self.wolp_action(new_state.detach().numpy()[i],proto_action.detach().numpy()[i])
            # print('x',x.shape)
            target_actions.append(x)
        # target_actions, action_ = self.wolp_action(new_state.detach().numpy(),proto_action.detach().numpy())
        # print('action',action)
        target_actions = T.tensor(np.array(target_actions), dtype=T.float).reshape(100,1)
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

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_change = 0 if action==2 or action==0 else 1
        neighbours = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
        lane = self.env.vehicle.target_lane_index[2] if isinstance(self.env.vehicle, ControlledVehicle) \
            else self.env.vehicle.lane_index[2]
        # if self.env.vehicle.crashed:
        #     reward = self.env.config["collision_reward"]*100
        # else:
        #     reward = 10
                # + self.env.RIGHT_LANE_REWARD * lane*10 / max(len(neighbours) - 1, 1) \
        scaled_speed = utils.lmap(self.env.vehicle.speed, self.env.config["reward_speed_range"], [0, 1])
        # print(scaled_speed, self.env.config["reward_speed_range"])
        # print(scaled_speed)
        # print(self.env.RIGHT_LANE_REWARD*lane)
        # print(self.env.config["collision_reward"])
        crashed = -1 if self.env.vehicle.crashed else 1
        reward = \
            + 0.05*lane_change \
            + 0.2 * lane / max(len(neighbours) - 1, 1) \
            + 0.65 * scaled_speed \
            + crashed*1.1

        # print(reward)
        # reward = self.env.config["collision_reward"] if self.env.vehicle.crashed else reward
        reward = utils.lmap(reward,
                          [-2, 2],
                          [-1, 1])
        # print(reward)
        # reward = utils.lmap(reward,
        #                 [self.env.config["collision_reward"]*5, (self.env.HIGH_SPEED_REWARD + self.env.RIGHT_LANE_REWARD)*5],
        #                 [0, 5])
        # print(reward)
        # print(self.env.vehicle.speed, reward)
        reward = 0 if not self.env.vehicle.on_road else reward
        return reward