import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNAgent:
    def __init__(self , model):
        self.model = model

    def get_actions(self , observation):
      # observation shape is (N, 4)
      qvals = self.model(observation)

      # q_vals shape (N, 2)

      return q_vals.max(-1)

class Model:
    def __init__(self , obs_shape , num_actions):
        self.obs_shape = obs_shape
        self.num_actions = num_actions

