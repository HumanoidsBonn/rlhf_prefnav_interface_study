import os.path

import numpy as np
import torch
from gym.core import RewardWrapper

from pd_morl_jorge.demo_tools.reward_model import *


class RewardModelWrapper(RewardWrapper):
    """
    Saves the last state to a separate variable.
    """
    def __init__(
            self,
            env,
            args,
            reward_model_path=None,
            reward_scale=1,
            reward_offset=0,
            concat=True,
            sum_id=0,
    ):
        super(RewardModelWrapper, self).__init__(env)
        self.reward_model = globals()[args.reward_model_class](args)
        reward_model_path = os.path.join(".", reward_model_path)
        self.reward_model.load_state_dict(torch.load(reward_model_path))
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset
        self.concat = concat
        self.sum_id = sum_id

        self.observation_last = None
        self.action_last = None

    def reward(self, reward):
        with torch.no_grad():  # Disable gradient calculation for inference
            model_reward = self.reward_model(self.observation_last, self.action_last).squeeze()

            if model_reward.requires_grad:
                model_reward = model_reward.detach()

            model_reward = model_reward.numpy()
            model_reward = model_reward * self.reward_scale + self.reward_offset

        if isinstance(reward, (float, int)):
            reward += model_reward
        elif isinstance(reward, (list, np.ndarray)):
            reward = np.array(reward)
            if self.concat:
                model_reward = np.expand_dims(model_reward, axis=0)
                reward = np.concatenate([model_reward, reward])
            else:
                reward[self.sum_id] = reward[self.sum_id] + model_reward
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.observation_last = torch.from_numpy(observation).float().unsqueeze(0)
        if isinstance(action, torch.Tensor):
            self.action_last = action.float().unsqueeze(0)
        else:
            self.action_last = torch.from_numpy(action).float().unsqueeze(0)
        return observation, self.reward(reward), done, info




