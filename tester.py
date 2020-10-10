import gym
import sys

args = sys.argv
env = gym.make(args[1])

print(env.action_space.shape)
print(env.observation_space.shape)