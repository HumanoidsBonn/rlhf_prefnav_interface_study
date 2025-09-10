from abc import ABC, abstractmethod
import numpy as np


class MotionSimulator2D(ABC):
    def __init__(self, time_step=0.1):
        self.time_step = time_step
        self.current_time = 0.0

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def get_current_position(self):
        raise NotImplementedError


class MotionSimulator2DWaypoints(MotionSimulator2D):
    def __init__(self, waypoints=None, velocities=None, time_step=0.1, wait_time=0.0):
        super(MotionSimulator2DWaypoints, self).__init__(time_step)

        self.waypoints = np.array(waypoints) if waypoints is not None else np.array([])
        self.velocities = np.array(velocities) if velocities is not None else np.array([])

        self.current_position = self.waypoints[0] if waypoints is not None else np.array([0.0, 0.0])
        self.current_velocity = self.velocities[0] if velocities is not None else 0.0

        self.current_index = 0
        self.wait_time = wait_time

        self.reset()

    def reset(self):
        self.current_time = 0.0
        self.current_index = 0
        if self.waypoints.size > 0:
            self.current_position = self.waypoints[0]
            self.current_velocity = self.velocities[0]
        else:
            self.current_position = np.array([0.0, 0.0])
            self.current_velocity = 0.0

    def set_waypoints(self, waypoints, velocities):
        self.waypoints = np.array(waypoints)
        self.velocities = np.array(velocities)
        self.reset()

    def step(self):
        # Wait for the specified wait time before starting the motion
        if self.current_time < self.wait_time:
            self.current_time += self.time_step
            return self.current_position, self.current_velocity

        if self.waypoints.size == 0 or self.velocities.size == 0:
            return self.current_position

        while self.current_index < len(self.waypoints) - 1:
            start_point = self.waypoints[self.current_index]
            end_point = self.waypoints[self.current_index + 1]
            segment_vector = end_point - start_point
            segment_length = np.linalg.norm(segment_vector)
            segment_unit_vector = segment_vector / segment_length

            start_velocity = self.velocities[self.current_index]
            end_velocity = self.velocities[self.current_index + 1]
            velocity_difference = end_velocity - start_velocity

            distance_traveled = np.linalg.norm(self.current_position - start_point)
            segment_fraction = distance_traveled / segment_length
            self.current_velocity = start_velocity + velocity_difference * segment_fraction

            max_distance = self.current_velocity * self.time_step
            distance_to_next_point = np.linalg.norm(self.current_position - end_point)

            if max_distance < distance_to_next_point:
                self.current_position = self.current_position + segment_unit_vector * max_distance
                self.current_time += self.time_step
                return self.current_position, self.current_velocity
            else:
                self.current_index += 1
                if self.current_index < len(self.waypoints) - 1:
                    self.current_position = self.waypoints[self.current_index]
                    self.current_velocity = self.velocities[self.current_index]
                else:
                    self.current_position = self.waypoints[-1]
                    self.current_velocity = self.velocities[-1]
                    break

        return self.current_position, self.current_velocity

    def get_current_position(self):
        return self.current_position