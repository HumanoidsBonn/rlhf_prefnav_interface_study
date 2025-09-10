from igibson.sensors.sensor_noise_base import BaseSensorNoise
from typing import List


class NoiseHandler(List[BaseSensorNoise]):
    "List of noise models. Calling add_noise() will iterate trough them."
    def add_noise(self, obs, aux=None):
        for noise_model in self:
            if noise_model.add_noise.__defaults__ is not None:
                obs = noise_model.add_noise(obs, aux)
            else:
                obs = noise_model.add_noise(obs)
        return obs
