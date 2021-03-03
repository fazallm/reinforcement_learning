import highway_env
import gym
import numpy as np
from ppo_highway import PPO, Memory


if __name__=='__main__':

    env_name = "highway-v0"
        # creating environment
    env = gym.make(env_name)
    state_dim = env.reset().reshape(-1).shape[0]
    action_dim = 5
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 5000        # max training episodes
    max_timesteps = 1000        # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 1000      # update policy every n timesteps
    lr = 0.00001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    score_history=[]
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, 'tmp/ppo-cont-new/eps-{}'.format(eps_clip))
    
    # logging variables
    
    avg_length = 0
    timestep = 0
    ppo.load()
    scores=[]
    for i in range(max_episodes):
        score = 0
        obs = env.reset()
        t=0
        while True:
            obs = obs.reshape(-1)
            action, log_prob = ppo.policy.act(obs)
            new_obs, rewards, terminal, info = env.step(action)
            memory.remember(obs, action, log_prob, rewards, terminal)
            obs = new_obs
            score+=rewards
            t+=1
            env.render()
            if terminal:
                break
        # if i%1==0:
        ppo.update(memory)
        memory.clear_memory()

        if i%25==0:
            ppo.save()
        scores.append(score)
        print('episode ', i, 'score %.2f' % score,
                'trailing 50 games avg %.3f' % np.mean(scores[-50:]),
                'finished after ', t, ' episode')
    env.close()