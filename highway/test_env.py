import highway_env
import gym
import numpy as np



env = gym.make('parking-v0')

print(env.observation_space)
obs = env.reset()
for _ in range(100):
    print(obs)
    action = env.action_space.sample()
    print(action)
    obs, rewards, terminate, something = env.step(action)
    print(obs,rewards, terminate, something)
    env.render()
env.close()

