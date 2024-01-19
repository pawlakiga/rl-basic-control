from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym 
import torch

class BasicEnvironment(gym.Env): 
    metadata = {"render_modes:" : []}

    def __init__(self, m = 1, d = 1, start_state = None, desired_state = 0.8, Ts = 0.1, render_mode = None, thresholds = [0.001, 0.001], max_episode_len = 30, Ki = 0) -> None:
        super().__init__()
        self.d = d 
        self.m = m
        self.start_state = start_state
        self.cur_state = start_state
        self.desired_state = desired_state
        self.Ts = Ts
        self.observation_space = gym.spaces.Box(low = 0, high = 10, dtype = float)
        self.action_space = gym.spaces.Box(low = 0, high = 1, dtype = float)
        self.render_mode = render_mode
        self.timestep = 0
        self.max_episode_len = max_episode_len
        self.thresholds = thresholds
        self.last_reward = 0
        self.int_reward = 0
        self.rewards_history = []
        self.Ki = Ki


    def reset(self, seed = None):
        super().reset(seed=seed)
        if self.start_state is None : 
            self.cur_state = self.observation_space.sample()
        else: 
            self.cur_state = np.array([self.start_state])
        self.timestep = 0
        self.last_reward = 0
        self.int_reward = 0
        self.rewards_history = []
        return self.cur_state, False

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_state = self.cur_state * (1 - self.d * self.Ts/self.m) + self.Ts / self.m * action
        self.timestep += 1
        done_state = self.done(new_state=new_state)
        self.cur_state = new_state
        self.last_reward =  self.reward_fun(action)
        # self.rewards_history.append(self.reward_fun(action))
        # self.last_reward = np.sum(self.rewards_history[:-1] * self.Ts + self.rewards_history[-1] )
        return self.cur_state, self.last_reward, done_state

    def done(self, new_state): 
        return self.timestep > self.max_episode_len
        # return ((np.abs(new_state - self.desired_state) <= self.thresholds[0]) and (np.abs(new_state - self.cur_state) <= self.thresholds[1])) or self.timestep > self.max_episode_len
    
    def reward_fun(self, action): 
        step_reward = - np.abs(self.desired_state  - self.cur_state) 
        self.int_reward += step_reward * self.Ts
        return step_reward + self.int_reward * self.Ki
        # return - np.abs(self.desired_state  - self.cur_state)  
      
    def close(self):
        return super().close()
        
class DelayedEnvironment(BasicEnvironment):
    def __init__(self, m=1, d=1, start_state=None, desired_state=0.8, Ts=0.1, render_mode=None, thresholds=[0.001, 0.001], max_episode_len=30, delay = 0) -> None:
        super().__init__(m, d, start_state, desired_state, Ts, render_mode, thresholds, max_episode_len)
        self.delay = delay
        self.actions_queue = []

    def step(self, action) : 
    
        if self.timestep < self.delay: 
            delayed_action = 0
            self.actions_queue.append(action)
        else: 
            if self.delay == 0:
                delayed_action = action
            else : 
                delayed_action = self.actions_queue.pop(0)
                self.actions_queue.append(action)
        # print(f"At timestep {self.timestep} actions queue {self.actions_queue}")
        return super().step(delayed_action)
    
    def reset(self):
        self.actions_queue = []
        return super().reset()    



    
