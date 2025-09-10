import cv2
import numpy as np
import pybullet as p
from transforms3d.quaternions import quat2mat
import numpy as np

from igibson.sensors.vision_sensor import VisionSensor
from igibson.utils.constants import OccupancyGridState

from gibson2.sensors.noise.depth_gaussian_noise import DepthGaussianNoise 
from gibson2.sensors.noise.depth_dropout_noise import DepthDropoutNoise
from gibson2.sensors.noise.noise_handler import NoiseHandler

class RGBD_sensor(VisionSensor):

    def __init__(self, env, modalities,):
        print(modalities)
        super(VisionSensor, self).__init__(env, modalities)

        # Noise: Dropout
        self.noise_model_dropout = DepthDropoutNoise(env, modalities)
        
        # Dropout configuration
        self.noise_model_dropout.set_distance_based(True) # True or False
        self.noise_model_dropout.set_dilate(False) # True or False
        self.noise_model_dropout.set_min_prob(0.1) # (0,1), min prob should be less than max prob
        self.noise_model_dropout.set_max_prob(0.97) # (0,1)

        # Noise: Gaussian
        self.noise_model_gaussian = DepthGaussianNoise(env)

        # Gaussian Configuration
        self.noise_model_gaussian.set_distance_based(True) # True or False
        self.noise_model_gaussian.set_const_std(0.01) # used when distance based is False
        self.noise_model_gaussian.set_max_dev(0.05) # [0,1], 
        self.noise_model_gaussian.set_min_dev(0.01) # [0,1]

        self.noise_model = NoiseHandler([
            self.noise_model_gaussian,
            self.noise_model_dropout,
        ])