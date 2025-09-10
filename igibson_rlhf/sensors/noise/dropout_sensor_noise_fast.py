import numpy as np

from igibson.sensors.sensor_noise_base import BaseSensorNoise
from igibson.sensors.dropout_sensor_noise import DropoutSensorNoise


class DropoutSensorNoiseFast(DropoutSensorNoise):
    """
    Naive dropout sensor noise model
    """

    def __init__(self, env):
        super(DropoutSensorNoise, self).__init__(env)
        self.noise_rate = 0.0
        self.noise_value = 1.0

    def add_noise(self, obs):
        """
        Add naive sensor dropout to perceptual sensor, such as RGBD and LiDAR scan

        :param sensor_reading: raw sensor reading, range must be between [0.0, 1.0]
        :param noise_rate: how much noise to inject, 0.05 means 5% of the data will be replaced with noise_value
        :param noise_value: noise_value to overwrite raw sensor reading
        :return: sensor reading corrupted with noise
        """
        if self.noise_rate <= 0.0:
            return obs

        assert len(obs[(obs < 0.0) | (obs > 1.0)]) == 0, "sensor reading has to be between [0.0, 1.0]"

        mask = np.random.uniform(size=obs.shape)
        obs[mask > 1.0 - self.noise_rate] = self.noise_value
        return obs
