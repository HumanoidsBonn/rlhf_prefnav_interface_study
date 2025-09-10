import cv2
import numpy as np
import pybullet as p
from transforms3d.quaternions import quat2mat
import numpy as np
from igibson_rlhf.sensors.noise.dropout_sensor_noise_fast import DropoutSensorNoiseFast
from igibson_rlhf.sensors.noise.gaussian_noise import GaussianSensorNoise
from igibson.sensors.scan_sensor import ScanSensor
from igibson.utils.constants import OccupancyGridState

from igibson_rlhf.sensors.noise.noise_handler import NoiseHandler


class ScanSensorRplidar(ScanSensor):
    """
    1D LiDAR scanner sensor and occupancy grid sensor
    """

    def __init__(self, env, modalities, rear=False):
        super(ScanSensorRplidar, self).__init__(env, modalities, rear)

        self.scan_noise_rate_gaussian = self.config.get("scan_noise_rate_gaussian", 0.0)

        # Noise: Dropout
        self.noise_model_dropout = DropoutSensorNoiseFast(env)
        self.noise_model_dropout.set_noise_rate(self.scan_noise_rate)
        self.noise_model_dropout.set_noise_value(1.0)

        # Noise: Gaussian
        self.noise_model_gaussian = GaussianSensorNoise(env)
        self.noise_model_gaussian.set_noise_rate(self.scan_noise_rate_gaussian)

        self.noise_model = NoiseHandler([
            self.noise_model_gaussian,
            self.noise_model_dropout,
        ])

    def get_obs(self, env):
        """
        Get current LiDAR sensor reading and occupancy grid (optional)

        :return: LiDAR sensor reading and local occupancy grid, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        if self.laser_link_name not in env.robots[0].links:
            raise Exception(
                "Trying to simulate LiDAR sensor, but laser_link_name cannot be found in the robot URDF file. Please add a link named laser_link_name at the intended laser pose. Feel free to check out assets/models/turtlebot/turtlebot.urdf and examples/configs/turtlebot_p2p_nav.yaml for examples."
            )
        laser_position, laser_orientation = env.robots[0].links[self.laser_link_name].get_position_orientation()
        angle = np.arange(
            -laser_angular_half_range / 180 * np.pi,
            laser_angular_half_range / 180 * np.pi,
            self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays,
        )
        unit_vector_local = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)  # Speedup of factor 150
        transform_matrix = quat2mat(
            [laser_orientation[3], laser_orientation[0], laser_orientation[1], laser_orientation[2]]
        )  # [x, y, z, w]
        unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

        start_pose = np.tile(laser_position, (self.n_horizontal_rays, 1))
        start_pose += unit_vector_world * self.min_laser_dist
        end_pose = laser_position + unit_vector_world * self.laser_linear_range
        results = p.rayTestBatch(start_pose, end_pose, numThreads=6)  # numThreads = 6

        # hit fraction = [0.0, 1.0] of self.laser_linear_range
        results = [sub_tuple[2] for sub_tuple in results]
        hit_fraction = np.array(results) # Speedup
        hit_fraction = self.noise_model.add_noise(hit_fraction)
        scan = np.expand_dims(hit_fraction, 1)

        state = {}
        state["scan" if not self.rear else "scan_rear"] = scan.astype(np.float32)
        if "occupancy_grid" in self.modalities:
            state["occupancy_grid"] = self.get_local_occupancy_grid(scan)
        return state


