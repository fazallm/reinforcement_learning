from ddpg_agent import Agent
import gym
import numpy as np
import matplotlib.pyplot as plt 

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

if __name__=='__main__':
    env = gym.make('LunarLanderContinuous-v2')
    # print(env.actions())
    # pass
    agent = Agent(alpha=25e-3, beta=25e-3, tau=0.005, env=env,
                batch_size=64, gamma=0.99)

    agent.load_models()
    np.random.seed(0)

    score_history = []
    for i in range(200):
        obs = env.reset()
        done = False
        score = 0
        step = 0
        while not done:
            step+=1
            # print(obs)
            act = agent.choose_action(obs)
            # print(act)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            env.render()
        score_history.append(score)

        # if i % 25 == 0:
        #     agent.save_models()

        print('episode ', i, 'score %.2f' % score,
            'trailing 128 games avg %.3f' % np.mean(score_history[-128:]),
            'finished after ', step, ' episode')
    env.close()
    agent.save_models()
    filename = 'MountainCar-alpha000025-beta00025-400-300.png'
    plotLearning(score_history, filename, window=100)
