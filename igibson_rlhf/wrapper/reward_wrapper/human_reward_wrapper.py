import os.path

import numpy as np
import torch
from gym.core import RewardWrapper
import torch
from torch import nn
#from pd_morl_jorge.demo_tools.reward_model import *


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        if j < len(sizes) - 2:
            layers += [nn.Dropout(0.5 if j > 0 else 0.2)]
    return nn.Sequential(*layers)


class HumanRewardNetwork(nn.Module):
    def __init__(self, obs_size, hidden_sizes=(64, 64)):
        super(HumanRewardNetwork, self).__init__()

        self.linear_relu_stack = mlp([obs_size] + list(hidden_sizes) + [1], activation=nn.LeakyReLU)
        self.tanh = nn.Tanh()

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return self.tanh(logits)


class HumanRewardModelWrapper(RewardWrapper):
    """
    Saves the last state to a separate variable.
    """
    def __init__(
            self,
            env,
            reward_model_path=None,
            reward_scale=1,
            reward_offset=0,
            reward_scale_old=1,
            reward_model_balance=1,
            concat=True,
            sum_id=0,
    ):
        super(HumanRewardModelWrapper, self).__init__(env)
        self.reward_model = HumanRewardNetwork(obs_size=68, hidden_sizes=[256,256,256])
        reward_model_path = os.path.join(".", reward_model_path)
        self.reward_model.load_state_dict(torch.load(reward_model_path))
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset
        self.reward_scale_old = reward_scale_old
        self.concat = concat
        self.sum_id = sum_id
        self.reward_model_balance = reward_model_balance

        self.observation_last = None
        self.action_last = None

    def reward(self, reward):
        raise NotImplementedError()

    def step(self, action):
        observation, reward_old, done, info = self.env.step(action)

        obs_tensor = torch.tensor(observation["task_obs"], dtype=torch.float32)
        obs_scan = torch.tensor(observation["scan"], dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        traj_tensor = torch.cat((obs_tensor, obs_scan, action), dim=0)

        reward_old = reward_old * self.reward_scale_old
        if not (self.reward_scale == 0 or self.reward_model_balance == 0):
            with torch.no_grad():
                reward_pbrl = self.reward_model(traj_tensor) * self.reward_scale + self.reward_offset
                reward_pbrl = reward_pbrl[0].item()


                reward = self.reward_model_balance * reward_pbrl + (1 - self.reward_model_balance) * reward_old
        else:
            reward = reward_old
        return observation, reward, done, info




