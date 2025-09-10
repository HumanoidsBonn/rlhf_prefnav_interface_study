from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson_rlhf.rewards.timeout_reward import TimeoutReward
from igibson_rlhf.rewards.continous_reward import ContinuousReward
from igibson_rlhf.rewards.cycle_reward import CycleReward
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d, restoreState
from igibson_rlhf.objects.pedestrian import Pedestrian
from igibson.objects.cube import Cube
from igibson.objects.visual_marker import VisualMarker


import numpy as np

import copy
import pybullet as p
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.timeout import Timeout

from igibson_rlhf.termination_conditions.out_of_bound import OutOfBound


class PointNavCenterObstacle(PointNavRandomTask):
    def __init__(self, env):
        super(PointNavCenterObstacle, self).__init__(env)
        self.number_of_boxes = env.config.get("number_of_boxes", 4)
        self.num_pedestrians = self.config.get(
            'num_pedestrians', 1)
        self.pedestrian_pos_noise = self.config.get(
            'pedestrian_pos_noise', 1.0)
        self.pedestrians = self.load_pedestrians(env)
        self.hidden_pedestrians = False
        self.save_id = None
        self.env = env
        self.avoid_poses = list()
        self.task_resets = 0

        self.goal_format = self.config.get("goal_format", "polar")
        self.reward_functions = [
            CollisionReward(self.config),
            PointGoalReward(self.config),
            TimeoutReward(self.config),
            ContinuousReward(self.config),
            CycleReward(self.config)
        ]
        self.cubes = self.load_boxes(env)
        # self.visual_markers_red = self.load_markers(env, color=[1,0,0,1])
        # self.visual_markers_blue = self.load_markers(env, color=[0,0,1,1])

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        # Direction to goal
        # =================================================================================
        goal_pose = self.global_to_local(env, self.target_pos)[:2]
        if self.goal_format == "polar":
            goal_pose = np.array(cartesian_to_polar(goal_pose[0], goal_pose[1]))
            # goal_pose = np.array([goal_pose[1]])  # ditch distance, only angle

        # Direction to human
        # =================================================================================
        assert len(self.pedestrians) in [1], "Task observation space designed only for one human!"
        ped = self.pedestrians[0]
        human_pose = self.global_to_local(env, ped.get_position())[:2]
        if self.goal_format == "polar":
            human_pose = np.array(cartesian_to_polar(human_pose[0], human_pose[1]))

        # Merging
        # =================================================================================
        task_obs = np.concatenate((
            goal_pose,
            human_pose,
        ))

        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(task_obs, [linear_velocity, angular_velocity])

        return task_obs

    def reset(self, env):
        self.reset_agent(env)
        self.reset_pedestrians(env)
        self.reset_scene(env)
        self.save_id = p.saveState()
        self.reset_for_episode(env)

        if self.save_id is not None:
            p.removeState(self.save_id)
            self.save_id = None

        self.save_id = p.saveState()
        self.task_resets += 1

    def reset_for_episode(self, env):
        self.reset_variables(env)
        for termination_condition in self.termination_conditions:
            termination_condition.reset(self, env)
        for reward in self.reward_functions:
            reward.reset(self, env)

    def reset_scene(self, env):
        ret_value = super().reset_scene(env)
        # print("reset cubes")
        self.cube_pos = list()
        for c, cube in enumerate(self.cubes):
            pos = self.get_position_for_box(env)
            # print("{} {}".format(c, pos))
            cube.set_position(pos)
            self.cube_pos.append(pos)
        return ret_value
    
    def reset_agent(self, env, disable_sample=False, demo_reset_call=False):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """

        self.reset_agent_poses(env)
        # We need to land the agent once more, as the state has been reset.
        env.land(env.robots[0], self.initial_pos, self.initial_orn)

    def reset_pedestrians(self, env):
        """
        Reset the poses of pedestrians to have no collisions with the scene or the robot and set waypoints to follow

        :param env: environment instance
        """
        for ped_id, (ped,) in enumerate(zip(self.pedestrians,)):
            ped_initial_pos = self.sample_pedestrian_pose(env, ped_id)
            ped.dynamic = False

            ped_initial_orn = p.getQuaternionFromEuler(ped.default_orn_euler)

            ped.set_position(ped_initial_pos)
            ped.set_orientation(ped_initial_orn)

    def sample_pedestrian_pose(self, env, ped_id, avoid_pose=None, independent_save_state=False):
        """
        Sample a new initial position for pedestrian with ped_id.
        The inital position is sampled randomly until the position is
        at least |self.orca_radius| away from all other pedestrians' initial
        positions and the robot's initial position.
        """
        remove_state = False
        if self.save_id is None or independent_save_state:
            state_id = p.saveState()
            remove_state = True
        else:
            state_id = self.save_id

        success = False
        max_trials = 200
        for i in range(max_trials):

            # Get pose just in between robot start and goal with some random noise.
            # ======================================================================
            initial_pos = (self.target_pos - self.initial_pos) / 2 + self.initial_pos
            pos_noise = np.random.normal(0.0, self.pedestrian_pos_noise, initial_pos.shape)
            pos_noise[2] = 0.0
            initial_pos += pos_noise

            initial_pos_w_z = copy.deepcopy(initial_pos)
            initial_pos_w_z[2] = 0.05

            # Check if valid starting point!
            reset_success = env.test_valid_position(
                self.pedestrians[ped_id],
                initial_pos_w_z,
                orn=self.pedestrians[ped_id].default_orn_euler,
                ignore_self_collision=True
            )
            restoreState(state_id)

            if not reset_success:
                continue
            success = True
            break

        initial_pos[2] = 0.2

        # TODO Remove before flight?
        if remove_state:
            p.removeState(state_id)

        self.pedestrians[ped_id].pos = initial_pos
        return initial_pos

    def reset_agent_poses(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        env.robots[0].reset()
        max_trials = 200

        remove_state = False
        if self.save_id is None:
            state_id = p.saveState()
            remove_state = True
        else:
            state_id = self.save_id
        for j in range(max_trials):
            for i in range(max_trials):

                # Old sampling routine
                _, initial_orn, _ = self.sample_initial_pose_and_target_pos(env)

                # new sampling routing
                max_trials = 100
                if not self.env.evaluation:
                    for _ in range(max_trials):
                        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
                        dist = np.linalg.norm(initial_pos)
                        if self.target_dist_min < dist * 2 < self.target_dist_max:
                            break
                else:
                    def polar_to_cartesian(r, phi):
                        """Convert cartesian coordinate to polar coordinate"""
                        x = r * np.cos(phi)
                        y = r * np.sin(phi)
                        return x, y

                    dist = self.target_dist_evaluation
                    initial_pos = [*polar_to_cartesian(dist, np.random.uniform(0, 2 * np.pi)), 0.0]
                    initial_pos = np.array(initial_pos)
                target_pos = - initial_pos

                reset_success = env.test_valid_position(
                    env.robots[0], initial_pos, initial_orn, ignore_self_collision=True
                ) and env.test_valid_position(env.robots[0], target_pos, ignore_self_collision=True)

                restoreState(state_id)
                if reset_success:
                    break

            self.target_pos = target_pos
            self.initial_pos = initial_pos
            self.initial_orn = initial_orn

            if self.config.get("orientation_goal_sampling_confined", False):
                self.initial_orn = self.point_robot_towards_human(initial_pos)

            land_success = env.land(env.robots[0], self.initial_pos, self.initial_orn, verbose=False)
            if land_success:
                break

        if remove_state:
            p.removeState(state_id)

    def get_position_for_box(self, env):
        max_trials = 100
        min_distance_by_box_size = 0.75 + 0.5 + 0.5  # sqrt(half_extents) + turtlebot_diam + buffer
        for _ in range(max_trials):
            _, box_initial_pos = env.scene.get_random_point(floor=self.floor_num)
            box_initial_pos[2] = 0
            dist = np.linalg.norm(box_initial_pos)

            # check for overlap with agent goal
            if np.linalg.norm(box_initial_pos - self.target_pos) < min_distance_by_box_size:
                continue

            # check for overlap with agent start
            if np.linalg.norm(box_initial_pos - self.initial_pos) < min_distance_by_box_size:
                continue

            # check for overlap with pedestrian poses
            if np.linalg.norm(box_initial_pos - self.pedestrians[0].pos) < min_distance_by_box_size:
                continue

            # Box placement in half target range around zero
            if not (self.target_dist_min < dist * 2 < self.target_dist_max):
                continue

            # # Avoid boxes on path
            # if self.is_close_to_path(box_initial_pos, min_distance_by_box_size):
            #     continue

            break

        return box_initial_pos

    def is_close_to_path(self, sampled_pose, d_thresh):
        """
        Check if point C is closer than d meters to the line segment AB.

        Parameters:
        A, B, C: tuples or lists representing points in 2D space (x, y).
        d: float, the threshold distance.

        Returns:
        bool: True if C is closer than d meters to the line segment AB, False otherwise.
        """
        # Convert points to numpy arrays for vectorized operations
        A = np.array(self.initial_pos)
        B = np.array(self.target_pos)
        C = np.array(sampled_pose)

        # Vector AB and vector AC
        AB = B - A
        AC = C - A

        # To check if C projects onto the line segment, calculate the dot products
        dot_product1 = np.dot(AB, AC)
        dot_product2 = np.dot(AB, C - B)

        # If C's projection on AB is outside of the segment, return distance to the nearest endpoint
        if dot_product1 < 0:
            return np.linalg.norm(AC) < d_thresh
        if dot_product2 > 0:
            return np.linalg.norm(C - B) < d_thresh

        # Perpendicular distance from C to line AB
        area_of_triangle = np.abs(AB[0] * AC[1] - AB[1] * AC[0])
        AB_length = np.linalg.norm(AB)
        distance = area_of_triangle / AB_length

        return distance < d_thresh

    def point_robot_towards_human(self, initial_pose, target_pose=None):
        """based on the assumption that taget_pose is opposite of initial_pose"""
        x, y, _ = initial_pose
        r, phi = cartesian_to_polar(x, y)
        yaw_offset = np.random.uniform(-1 * np.pi, +1 * np.pi) / 360 * self.config.get("orientation_goal_range", 45)
        orientation_to_goal = [0, 0, phi + np.pi + yaw_offset]
        return orientation_to_goal

    def load_boxes(self, env, number_of_boxes=None):
        number_of_boxes = number_of_boxes or self.number_of_boxes
        cubes = []
        for _ in range(number_of_boxes):
            max_trials = 100
            pos = self.get_position_for_box(env)
            dim = [0.5, 0.5, 0.5]
            cube = Cube(pos, dim, mass=0)
            cube.load(env.simulator)
            cube.set_position(pos)
            cubes.append(cube)
        return cubes

    def load_markers(self, env, color=[1, 1, 1, 1]):
        markers = []
        for _ in range(100):
            max_trials = 100
            pos = [0, 0, -10]
            marker = VisualMarker(rgba_color=color, radius=0.1)
            marker.load(env.simulator)
            marker.set_position(pos)
            markers.append(marker)
        return markers

    def load_pedestrians(self, env):
        """
        Load pedestrians

        :param env: environment instance
        :return: a list of pedestrians
        """
        weg = 0  # 0.5
        scale = 0  # 2.
        pedestrians = []
        for i in range(self.num_pedestrians):
            ped = Pedestrian(
                style=1,
                pos=[weg + (i * scale), weg + (i * scale), 0.],
                exclude_from_physics=True,
                dynamic=True,
                class_id=398
            )

            env.simulator.import_object(ped)
            pedestrians.append(ped)

        return pedestrians

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return done: whether the episode has terminated
        :return info: additional info
        """
        done = False
        success = False
        collision = False
        timeout = False
        bounds = False

        for condition in self.termination_conditions:
            d, s = condition.get_termination(self, env)
            # if d:
            #     print(condition)
            done = done or d
            success = success or s
            if type(condition) is MaxCollision:
                collision = True if d else False
            if type(condition) is Timeout:
                timeout = True if d else False
            if type(condition) is OutOfBound:
                bounds = True if d else False



        info["done"] = done
        info["success"] = success
        info["goal_reached"] = success
        info["bumped"] = collision
        info["max_episode_steps_reached"] = timeout
        info["out_of_bounds"] = bounds

        # if done:
        #     print(info)

        return done, info