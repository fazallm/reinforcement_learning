from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gym
from stable_baselines3.common.callbacks import EveryNTimesteps, StopTrainingOnMaxEpisodes, CallbackList, CheckpointCallback
from stable_baselines3 import PPO, DDPG
import numpy as np
import matplotlib.pyplot as plt
import os
# from highway_env.envs.common.action import Action
# from stable_baselines3.common.vec_env import (
#     DummyVecEnv,
#     VecEnv,
#     VecNormalize,
#     VecTransposeImage,
#     is_vecenv_wrapped,
#     unwrap_vec_normalize,
# )
# from highway_env.vehicle.controller import ControlledVehicle
# from highway_env import utils


def runner(agent, episode, checkpoint, env):
    # scores = np.genfromtxt(checkpoint+'/data.csv', delimiter=',')
    # checkpoint2 = checkpoint+'2'
    custom_callback = LoggerCallback(episode, checkpoint=checkpoint)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpoint,
                                            name_prefix='rl_model')
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=episode, verbose=1)
    event_callback = EveryNTimesteps(n_steps=1, callback=custom_callback)
    # load = os.path.abspath(checkpoint+'/rl_model_676000_steps')
    # print(load)
    # agent = DDPG.load(load, env)
    callback_list = CallbackList([event_callback, checkpoint_callback, callback_max_episodes])
    # agent.learn(total_timesteps=100000000, callback=callback_list, reward_function=reward)
    agent.learn(total_timesteps=100000000, callback=callback_list)
    scores = custom_callback.rewards
    np.savetxt(checkpoint+'/data.csv', scores, delimiter=',')

    return scores

# def reward(new_env:DummyVecEnv,action: Action) -> float:
#         """
#         The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
#         :param action: the last action performed
#         :return: the corresponding reward
#         """
#         env = new_env.envs[0]
#         lane_change = 0 if action==2 or action==0 else 1
#         neighbours = env.road.network.all_side_lanes(env.vehicle.lane_index)
#         lane = env.vehicle.target_lane_index[2] if isinstance(env.vehicle, ControlledVehicle) \
#             else env.vehicle.lane_index[2]
#         # if env.vehicle.crashed:
#         #     reward = env.config["collision_reward"]*100
#         # else:
#         #     reward = 10
#                 # + env.RIGHT_LANE_REWARD * lane*10 / max(len(neighbours) - 1, 1) \
#         scaled_speed = utils.lmap(env.vehicle.speed, env.config["reward_speed_range"], [0, 1])
#         # print(scaled_speed, env.config["reward_speed_range"])
#         # print(scaled_speed)
#         # print(env.RIGHT_LANE_REWARD*lane)
#         # print(env.config["collision_reward"])
#         crashed = -1 if env.vehicle.crashed else 1
#         reward = \
#             + 0.05*lane_change \
#             + 0.2 * lane / max(len(neighbours) - 1, 1) \
#             + 0.65 * scaled_speed \
#             + crashed*1.1

#         # print(reward)
#         # reward = env.config["collision_reward"] if env.vehicle.crashed else reward
#         reward = utils.lmap(reward,
#                           [-2, 2],
#                           [-1, 1])
#         # print(reward)
#         # reward = utils.lmap(reward,
#         #                 [env.config["collision_reward"]*5, (env.HIGH_SPEED_REWARD + env.RIGHT_LANE_REWARD)*5],
#         #                 [0, 5])
#         # print(reward)
#         # print(env.vehicle.speed, reward)
#         reward = 0 if not env.vehicle.on_road else reward
#         return reward

class LoggerCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, episode, checkpoint, verbose=0, rewards=[]):
        super(LoggerCallback, self).__init__(verbose)
        self.max_episode = episode
        self.checkpoint=checkpoint
        self.rewards = rewards.tolist() if isinstance(rewards,(np.ndarray)) else rewards
        self.reward = 0
        self.count=0
        self.step = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.

        """
        # print(self.locals)
        self.locals["env"].render()
        # print(self.locals)
        done = None
        self.step+=1
        if 'PPO' in self.checkpoint:
            self.reward +=self.locals["rewards"][0]
            done = self.locals["dones"][0]
        else:
            self.reward +=self.locals["reward"][0]
            done = self.locals["done"][0]
        # print(self.reward)
        # done = self.locals["done"][0]
        if done:
            self.rewards.append(self.reward)
            self.count+=1
            print("Episode {}, Total rewards is {}, total {} steps".format(self.count, self.reward, self.step))
            self.reward=0
            self.step = 0
            
            
        if self.count==self.max_episode:
            print('Saving final model to: {}'.format(self.checkpoint))
            self.locals["self"].save(self.checkpoint+'/final')
        if self.count%25==0:
            np.savetxt(self.checkpoint+'/data.csv', self.rewards, delimiter=',')
            # print(self.rewards)
        # if self.count==10:
        #     self.locals["n_steps"]=100000
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

