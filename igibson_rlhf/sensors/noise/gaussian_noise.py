import numpy as np
from igibson.sensors.sensor_noise_base import BaseSensorNoise


class GaussianSensorNoise(BaseSensorNoise):
    """
    Naive dropout sensor noise model
    """

    def __init__(self, env):
        super(GaussianSensorNoise, self).__init__(env)
        self.mode = "multiplicative"  # (multiplicative, additive)
        self.noise_mu = 0.0
        self.noise_sigma = 0.0

    def set_noise_rate(self, noise_rate):
        """
        Set noise rate

        :param noise_rate: noise rate
        """
        self.noise_sigma = noise_rate

    def add_noise(self, obs):
        if self.noise_sigma <= 0.0:
            return obs

        noise = np.random.normal(loc=0, scale=self.noise_sigma, size=obs.shape)

        if self.mode == "multiplicative":
            obs = obs + obs * noise
        elif self.mode == "additive":
            obs = obs + noise
        else:
            raise RuntimeWarning("Noise mode unknown in GaussianSensorNoise")

        return obs