import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from networks import *
from ddpg_utils import *
from environment import *
import pickle
import os 
import tqdm

# Definition of the agent class
class DDPGAgent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.memory = ExperienceReplayBuffer(maximum_length=max_memory_size)
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample_batch(n = batch_size)
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(np.array(actions)).float()
        rewards = torch.tensor(np.array(rewards)).float()
        next_states = torch.tensor(np.array(next_states)).float()
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save(self, models_path) :
        if not os.path.isdir(models_path):
            os.makedirs(models_path)
        torch.save(self.actor, os.path.join(models_path, 'actor.pth'))
        torch.save(self.actor_target, os.path.join(models_path,'actor-target.pth'))
        torch.save(self.critic, os.path.join(models_path,'critic.pth'))
        torch.save(self.critic_target, os.path.join(models_path,'critic-target.pth'))

###############################################################################################################################################
############################## Functions for training and testing the agent ###################################################################
        

def train_agent(m, max_episode_len, Ki, n_episodes):
    print(f"In train agent")
    env = BasicEnvironment(m = m, max_episode_len = max_episode_len, Ki = Ki)
    print(f"Initialised environment")
    agent = DDPGAgent(env)
    print(f"Initialised agent")
    batch_size = 128
    rewards = []
    avg_rewards = []
    last_rewards = []
    episodes = tqdm.trange(n_episodes, desc = 'Training', leave = True)

    for episode in episodes:
        
        state, done = env.reset()
        state = np.array(state, dtype=float)
        episode_reward = 0
        t = 0
        while not done:
            action = agent.get_action(state)
            # print(f"At timestep {t} action is {action}\r")
            new_state, reward, done = env.step(action) 
            if done : 
                break
            new_state = np.array(new_state, dtype=float)
            action = np.array([action], dtype=float)
            reward = np.array(reward, dtype = float)
            agent.memory.append((state, action, reward, new_state, done))
            agent.memory.append((state, action, reward, new_state, done))
            
            if len(agent.memory) > batch_size:
                agent.update(batch_size)        
            
            state = new_state
            episode_reward += reward[0]
            t+= 1

        last_rewards.append(reward[0])
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
        episodes.set_description("episode: {} terminated after {} steps, last state {:4f}, reward: {}, average reward: {:2f}".format(episode,t, state[0], np.round(episode_reward, decimals=2), np.mean(rewards[-5:])))

        # print("episode: {} terminated after {} steps, last state {}, reward: {}, average reward: {}".format(episode,t, state[0], np.round(episode_reward, decimals=2), np.mean(rewards[-5:])))
        
        
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot(last_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['episode rewards', 'average episode rewards', 'last rewards in episode'])
    plt.grid()
    plt.show()
    return agent

def test_agent(m, delay, agent): 
    delayed_env = DelayedEnvironment(m = m, start_state=0.0, delay=delay, max_episode_len=30*(delay+1))
    ######################## Simulation of the agent #####################################
    states = []
    rewards = []
    cum_rewards = []
    actions = []
    state, done = delayed_env.reset()
    i = 0
    while not done: 
        action = agent.get_action(state)
        # if delayed_env.timestep < delayed_env.delay : 
        #     actions.append(0)
        # else : 
        #     actions.append(delayed_env.actions_queue[0])
        next_state, reward, done = delayed_env.step(action=action)
        if done : 
            print(f"Terminated at timestep {i} with current: {delayed_env.cur_state} and next: {next_state} and desired : {delayed_env.desired_state}")
            break
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if i == 0 : 
            cum_rewards.append(reward)
        else: 
            cum_rewards.append(cum_rewards[-1] + reward)
        i+=1
        state = next_state
    ######################## Simulation with a step function #############################
    states_model = []
    rewards_model = []
    cum_rewards_model = []
    state, done = delayed_env.reset()
    i = 0
    while not done: 
        action = 0.8
        next_state, reward, done = delayed_env.step(action=action)
        if done : 
            break
        states_model.append(state)
        rewards_model.append(reward)
        if i == 0 : 
            cum_rewards_model.append(reward)
        else: 
            cum_rewards_model.append(cum_rewards[-1] + reward)
        i+=1
        state = next_state

    f, ax = plt.subplots(1,3,figsize = (20,3))   
    ax[0].plot(states)
    ax[0].plot(states_model)
    ax[0].axhline(y = delayed_env.desired_state, xmax=len(states)-1, color = 'green', linestyle = '--')
    ax[0].grid()
    ax[0].legend(['agent', 'model','desired state'])
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('state')
    ax[1].plot(rewards)
    ax[1].plot(rewards_model)
    ax[1].legend(['agent', 'model'])
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('step reward')
    ax[1].grid()
    ax[2].plot(actions)
    ax[2].legend(['agent selected action'])
    ax[2].axhline(y = delayed_env.desired_state, xmax=len(states_model)-1, color = 'orange')
    ax[2].grid()
    plt.show()

    return states, states_model, rewards, rewards_model, actions
