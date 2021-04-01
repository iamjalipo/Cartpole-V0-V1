import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from random import sample

@dataclass
class Sarsd:
    state : Any
    action : int
    reward : float
    done : bool
    next_state : Any

class DQNAgent:
    def __init__(self , model):
        self.model = model

    def get_actions(self , observation):
      # observation shape is (N, 4)
      qvals = self.model(observation)

      # q_vals shape (N, 2)

      return q_vals.max(-1)

class Model(nn.Module):
    def __init__(self , obs_shape , num_actions):
        super(Model , self).__init__()
        assert len(obs_shape) == 1 , "this network only work for flat observation"
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0] , 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256 , num_actions)
            # we dont need activation , we reperesent real numbers
        )
        self.optim = optim.Adam(lr = 1e-4)

    def forward(self , x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self , buffer_size = 100000):
        self.buffer_size = buffer_size
        self.buffer = []

    def insert(self , sars):
        self.buffer.append(sars)
        self.buffer = self.buffer[-self.buffer_size:]

    def sample(self , num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer , num_samples)

def update_tgt_model(m , tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model , state_transitions , tgt):
    cur_state = torch.stack([s.state for s in state_transitions])
    rewards = torch.stack([s.reward for s in state_transitions])
    mask = torch.stack([0 if s.done else 1 for s in state_transitions])
    next_states = torch.stack([s.next_state for s in state_transitions])
    actions = [s.action for s in state_transitions]
    #not discount factor yet
    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1) #(N , num_actions)

    qvals = model(cur_state) #(N , num_action)
    one_hot_actions = F.one_hot(torch.LongTensor(actions , num_actions))

    rewards + qvals_next - qvlas*one_hot_actions


if __name__ == '__main__' :
    env = gym.make("CartPole-v1")
    last_observation = env.reset()

    m = Model(env.observation_space.shape , env.action_space.n)
    tgt = Model(env.observation_space.shape , env.action_space.n)
   

    rb = ReplayBuffer()

    #qvals = m(torch.Tensor(observation))
    #import ipdb ; ipdb.set_trace()
   
    try: 
        while True:
        #env.render()
        #time.sleep(0.1)
            action = env.action_space.sample()
        #env.action_space.n get number of action
        #env.observation_space.shape to get shape of observation
            observation, reward , done , info = env.step(action)
            
            rb.insert(Sarsd(last_observation,action, reward , observation , done))
            last_observation = observation

            if done:
                    observation= env.reset

            if len(rb.buffer) > 5000:
                import ipdb ; ipdb.set_trace()        
    
    except KeyboardInterrupt:
        pass

    env.close


