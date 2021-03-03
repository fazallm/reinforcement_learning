from ppo_highway import PPO
from ddpg_highway import WolpertingerAgent as Agent
import matplotlib.pyplot as plt
import numpy as np
import gym
import highway_env
from baseline_runner import runner as baseline_runner
from stable_baselines3 import PPO as PPObaseline, DDPG as DDPGbaseline
# from ddpg_highway import Agent, WolpertingerAgent

def plotLearning(scores, title, x=None, window=20):
    # print(len(scores))
    # length = len(scores)
    N = len(scores[0]["score"])
    # print(scores)
    if x is None:
        x = [i for i in range(N)]
    plt.title(title)
    plt.ylabel('Score')       
    plt.xlabel('Game')
    for i in scores:
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(i["score"][max(0, t-window):(t+1)])
        plt.plot(x, running_avg, label='experiment-{}'.format(i["name"]))
    plt.legend(loc='best')
    plt.savefig(title+'.png')
    plt.close()



if __name__=='__main__':
    # list_env = ["LunarLanderContinuous-v2", "HalfCheetah-v2", "HumanoidStandup-v2"]
    list_env = ["LunarLanderContinuous-v2"]
    episode = 1000
    version='v2'

    for env_name in list_env:
        env = gym.make(env_name)
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range":{
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                # "normalized": False,
                # "order": "sorted"
            },
            # "action":{
            #     "type": "ContinuousAction",
            #     "steering_range": [-0.1,0.1]
            # },
            "vehicles_density":1.3,
            "vehicles_count":20,
            "lanes_count": 2,
            "controlled_vehicles": 1,
            "duration":50,
            "collision_reward": -1,
            "reward_speed_range":[20,50],
            "offroad_terminal": True
        }
        # env.configure(config)
        env.reset()
        # env.config["vehicles_density"]=1.5
        # env.config["vehicles_count"]=30
        # # env.config["reward_speed_range"]=[30,60]
        # env.config["duration"]=100
        # env.config["collision_reward"]=-100
        # env.config["normalize"]=False
        np.random.seed(0)
        # agent = WolpertingerAgent(alpha=0.002, beta=0.002, tau=0.005, env=env, size=300000,
        #             batch_size=10, gamma=0.99, checkpoint="tmp/{}/{}/{}".format(env_name, version, "DDPG"))
        # agent = PPO(env, 32,0.0001, (0.99, 0.999),0.9, 3, 0.2, checkpoint="tmp/{}/{}/{}".format(env_name, version,"PPO"))
        # agent = PPObaseline('MlpPolicy', env)
        list_agent = [
            {
                "type": "own",
                "name": "PPO-my implementation",
                "checkpoint":"./tmp/{}/{}/{}".format(env_name, version, "PPO"),
                "agent": PPO(env, 128,0.0003, (0.99, 0.999),0.99, 5, 0.1, checkpoint="tmp/{}/{}/{}".format(env_name, version,"PPO"))
            },
            {
                "type": "baseline",
                "name": "PPO-baseline",
                "checkpoint": "./tmp/{}/{}/{}".format(env_name, version,"PPO-baseline"),
                "agent": PPObaseline('MlpPolicy', env)
            }
            # {
            #     "type": "own",
            #     "name": "DDPG-my implementation",
            #     "checkpoint": "./tmp/{}/{}/{}".format(env_name, version,"DDPG"),
            #     "agent": Agent(alpha=0.002, beta=0.002, tau=0.005, env=env, size=300000,
            #         batch_size=100, gamma=0.99, checkpoint="tmp/{}/{}/{}".format(env_name, version, "DDPG"))
            # }
            # {
            #     "type": "baseline",
            #     "name": "DDPG-baseline",
            #     "checkpoint": "./tmp/{}/{}/{}".format(env_name, version, "DDPG-baseline"),
            #     "agent": DDPGbaseline('MlpPolicy',env=env)
            # }
        ]
        # agent = {
        #         "type": "baseline",
        #         "name": "DQN-baseline",
        #         "checkpoint": "./tmp/{}/{}/{}".format(env_name, version,"DQN-baseline"),
        #         "agent": DQN('MlpPolicy', env)
        #     }
        scores = []
        # score = baseline_runner(agent["agent"], episode, agent["checkpoint"], env)
        for agent in list_agent:
            # print(agent["checkpoint"])
            if agent["type"]=='own':
                runner = agent["agent"]
                score = runner.runner(episode, render=True, train=True)
            else:
                # agent = agent["agent"]
                score = baseline_runner(agent["agent"], episode, agent["checkpoint"], env)
            result={
                "name":agent["name"],
                "score": score
            }
            scores.append(result)
            print(len(scores))
        # score = agent.runner(episode, render=True, train=True)
        # score = runner(agent, episode,"./tmp/{}/{}/{}".format(env_name, version,env_name), env)
        # score = runner(agent, 1000, "./tmp/{}/{}/{}".format(env_name, version,"PPO-baseline"), env)
        plotLearning(scores,  "./tmp/{}/{}/{}".format(env_name, version,'highway-continuous'))