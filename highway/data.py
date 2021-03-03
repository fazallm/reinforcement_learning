import matplotlib.pyplot as plt
import numpy as np

def plotLearning(scores, title, x=None, window=100):
    print(len(scores))
    length = len(scores)
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
    env_name='HalfCheetah-v2/v2'
    list_agent = [
        {
            "type": "own",
            "name": "PPO-my implementation",
            "checkpoint":"./tmp/{}/{}".format('highway-v0/v18', "PPO")
        },
        {
            "type": "baseline",
            "name": "PPO-baseline",
            "checkpoint": "./tmp/{}/{}".format('highway-v0/v4', "PPO-baseline")
        }
        # {
        #     "type": "own",
        #     "name": "DDPG-my implementation",
        #     "checkpoint": "./tmp/{}/{}".format(env_name, "DDPG")
        # },
        # {
        #     "type": "baseline",
        #     "name": "DDPG-baseline2",
        #     "checkpoint": "./tmp/{}/{}".format(env_name, "DDPG-baseline2")
        # }
    ]

    for agent in list_agent:
        score = np.genfromtxt('{}/data.csv'.format(agent["checkpoint"]), delimiter=',')
        score=score[:1000]
        agent["score"] = score

    plotLearning(list_agent, 'highway-v0 plot')
