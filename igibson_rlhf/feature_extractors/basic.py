
import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TaskObsAndScanExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(TaskObsAndScanExtractor, self).__init__(observation_space, features_dim=128)
        
        feature_size = 64
        self.observation_extractor = nn.Sequential(nn.Linear(observation_space.spaces["task_obs"].shape[0], feature_size), nn.ReLU(), nn.Linear(feature_size, feature_size), nn.ReLU())
        self.scan_extractor = nn.Sequential(nn.Linear(observation_space.spaces["scan"].shape[0], feature_size), nn.ReLU(), nn.Linear(feature_size, feature_size), nn.ReLU())
        self.final_layer = nn.Sequential(nn.Linear(2 * feature_size, 2*feature_size), nn.ReLU())

        # Update the features dim manually
        # self._features_dim = feature_size

    def forward(self, observations) -> th.Tensor:
        task_obs_act = self.observation_extractor(observations["task_obs"])
        scan_obs_act = self.scan_extractor(observations["scan"])
        tensor_merge = th.cat([task_obs_act, scan_obs_act],dim=-1)
        return self.final_layer(tensor_merge)