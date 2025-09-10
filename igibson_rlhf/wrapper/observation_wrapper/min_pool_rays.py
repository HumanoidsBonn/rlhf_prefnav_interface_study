from gym import ObservationWrapper
import numpy as np
import math
from gym.spaces import Box, Dict


class MinPoolRays(ObservationWrapper):
    obs_space = Dict({
            "task_obs": Box(shape=(6,), low=-np.inf, high=np.inf),
            "scan": Box(shape=(60,), low=0., high=1.)
    })
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Dict({
            "task_obs": Box(shape=(6,), low=-np.inf, high=np.inf),
            "scan": Box(shape=(60,), low=0., high=1.)
        })

    def get_min_pool(self, scan, window_size=12):
        window_mins = []
        num_windows = math.ceil(scan.shape[0]/window_size)
        for idx, _ in enumerate(range(num_windows)):
            window_start = window_size * idx
            window_end = window_size*(idx+1)
            window_mins.append(np.min(scan[window_start:window_end,:], axis=0))
        return np.array(window_mins).reshape((num_windows))

    def observation(self, obs):
        obs["scan"] = self.get_min_pool(obs["scan"])
        return obs

    def get_state(self):
        return self.observation(self.env.get_state())