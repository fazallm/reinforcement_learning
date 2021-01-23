from ppo_agent_cont import PPO, Memory
import gym
import matplotlib.pyplot as plt
import numpy as np
import time

def plotLearning(scores, filename, x=None, window=5):   
    length = len(scores)
    N = len(scores[0])
    # print(scores)
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')
    for i in range(length):
        running_avg = np.empty(N)
        for t in range(N):
	        running_avg[t] = np.mean(scores[i][max(0, t-window):(t+1)])
        plt.plot(x, running_avg, label='experiment-{}'.format(i+1))
    plt.legend(loc='best')
    plt.savefig(filename)


if __name__=='__main__':


    env_name = "LunarLanderContinuous-v2"
        # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000        # max training episodes
    max_timesteps = 1000        # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 4000      # update policy every n timesteps
    lr = 0.0003
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 80                # update policy for K epochs
    eps_clip = 0.1              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    score_history=[]

    for epoch in range(1):
        memory = Memory()
        goodMemory = Memory()
        ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, 'tmp/ppo-cont-new-model/eps-{}'.format(eps_clip))
        
        # logging variables
        
        avg_length = 0
        timestep = 0
        # ppo.load()
        scores=[]
        # training loop
        step = 0
        for i_episode in range(1, max_episodes+1):
            state = env.reset()
            running_reward = 0.0
            
            for t in range(max_timesteps):
                timestep += 1
                
                # Running policy_old:
                action, log_prob = ppo.select_action(state)
                

                new_state, reward, done, info = env.step(action.cpu().detach().numpy())
                # time.sleep(0.01)
                # Saving reward and is_terminal:
                # state = torch.from_numpy(state).float().to(device)
                # states.append(state)
                # actions.append(action)
                # log_probs.append(log_prob)
                # rewards.append(reward)
                # is_terminals.append(done)

                memory.remember(state, action, log_prob, reward, done)
                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    # timestep = 0
                state = new_state
                running_reward += reward
                if render:
                    env.render()
                if done:
                    break
            # memory.remember(states, actions, log_probs, rewards, is_terminals)
            # if running_reward>0:
            #     goodMemory.remember(states, actions, log_probs, rewards, is_terminals)
            avg_length += t
            scores.append(running_reward)
            if i_episode%20==0:
                print('episode ', i_episode, 'score %.2f' % running_reward,
                    'trailing 20 games avg %.3f' % np.mean(scores[-20:]),
                    'finished after ', t, ' episode')
            if i_episode%25==0:
                ppo.save()
            # scores.append(running_reward)
            running_reward = 0.0
            
        # print('score is {}'.format(scores))
        score_history.append(scores)
        # eps_clip+=0.1
    env.close()
    filename = 'LunarLander-ppo-beta0.9-gamma0.9-epsilon0.2-400-300.png'
    plotLearning(score_history, filename, window=10)