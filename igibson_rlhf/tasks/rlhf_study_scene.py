import copy
import os

import pybullet_data
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.utils.constants import SemanticClass
from igibson_rlhf.objects.visual_marker_hrl import VisualMarkerHRL
from igibson_rlhf.objects.pedestrian import Pedestrian

from igibson_rlhf.rewards.timeout_reward import TimeoutReward
from igibson_rlhf.rewards.continous_reward import ContinuousReward
from igibson_rlhf.rewards.cycle_reward import CycleReward
from igibson_rlhf.rewards.jerk_reward import JerkReward
from igibson_rlhf.objects.cube import Cube


import numpy as np
import pybullet as p
from igibson.utils.utils import restoreState

from igibson_rlhf.tasks.point_nav_center_obstacle import PointNavCenterObstacle


class RLHFStudyScene(PointNavCenterObstacle):
    def __init__(self, env):
        self.config_room = {
            "width": 8,
            "length": 11,
            "height": 2,
            "wall_thickness": 0.2,
            "color_wall": [0.9, 0.9, 0.9, 1],
            "color_obstacles": [0.5, 0.5, 0.5, 1],
            "cube_large_number": 4,
            "cube_small_number": 4,
            "cube_large_dim": [0.4, 0.8, 0.6],
            "cube_small_dim": [0.15, 0.15, 2.0],
            "randomization": {
                "cubes_large": {},
                "cubes_small": {},
            }
        }

        super(RLHFStudyScene, self).__init__(env)
        self.avoid_poses_dist = 0.6

        self.cubes_small = list()
        self.cubes_large = list()
        self.walls = list()
        self.static_obstacles = list()
        self.human_pos_indicator = None

        self.load_room(env)

        self.dedicated_collision_objects = [
            *self.cubes_small,
            *self.cubes_large,
            *self.walls,
            *self.static_obstacles,
            *self.pedestrians,
        ]
        tmp = list()
        for obj in self.dedicated_collision_objects:
            tmp.extend(obj.get_body_ids())
        self.dedicated_collision_objects = tmp

        self.reward_functions = [
            CollisionReward(self.config),
            JerkReward(self.config),
            PointGoalReward(self.config),
            TimeoutReward(self.config),
            ContinuousReward(self.config),
            CycleReward(self.config)
        ]


        # ==============================================================================
        # Task specific observation space ingredients
        # ==============================================================================
        low = (
            # Goal
            # =======================================================
            np.array([0]),  # distance robot human [m]
            np.array([-1 * np.pi]),  # rel. angle robot goal [rad]
            # Human
            # =======================================================
            np.array([0]),  # distance robot human [m]
            np.array([-1 * np.pi]),  # rel. angle robot human [rad]
            # Velocities
            # =======================================================
            np.array([-1.0]),
            np.array([-3.14]),
            )
        high = (
            # Goal
            # =======================================================
            np.array([self.config.get("depth_high", 5.0)]),  # distance robot goal [m]
            np.array([+1 * np.pi]),  # rel. angle robot goal [rad]
            # Human
            # =======================================================
            np.array([self.config.get("depth_high", 5.0)]),  # distance robot human [m]
            np.array([+1 * np.pi]),  # rel. angle robot human [rad]
            # Velocities
            # =======================================================
            np.array([+1.0]),
            np.array([+3.14]),
            )
        self.task_observation_range = [low, high]

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
                class_id=398,
                scale=1.0,
                position_offset=[0,0,-0.2],
                visual_filename=os.path.join("./igibson_rlhf/objects/resources/mannequin/mannequin.obj")
            )

            env.simulator.import_object(ped)
            pedestrians.append(ped)

        return pedestrians

    def load_boxes(self, env, number_of_boxes=None):
        # Dummy function to prevent loading of boxes the old way
        return None

    def reset(self, env):
        self.hide_obstacles()
        self.reset_agent(env)
        self.save_id = None
        self.reset_pedestrians(env)
        if self.human_pos_indicator is not None:
            self.human_pos_indicator.set_position(self.pedestrians[0].pos)
        self.reset_scene(env)
        self.save_id = p.saveState()
        self.reset_for_episode(env)

        if self.save_id is not None:
            p.removeState(self.save_id)
            self.save_id = None

        self.save_id = p.saveState()
        self.task_resets += 1

    def reset_scene(self, env):
        self.randomize_room(env)
        ret_value = super(PointNavCenterObstacle, self).reset_scene(env)
        return ret_value
    
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
                initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

                # new sampling routing
                for _ in range(max_trials):
                    _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
                    dist = np.linalg.norm(initial_pos)

                    if not self.is_point_in_room(initial_pos):
                        continue

                    if self.target_dist_min < dist * 2 < self.target_dist_max:
                        break

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

            if not self.is_point_in_room(initial_pos):
                continue

            if np.linalg.norm(np.array(self.target_pos) - initial_pos) < self.target_dist_min:
                continue

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


    def check_avoid_poses(self, pose, buffer=0):
        """
        Checks a pose against a list of avoid poses.
        """
        success = True
        for avoid in self.avoid_poses:
            dist = np.linalg.norm(pose[:2] - avoid[:2])
            if self.avoid_poses_dist + buffer > dist:
                success = False
                break
        return success

    def get_position_for_box(self, env, box_max):
        max_trials = 100
        min_distance_by_box_size = box_max + 0.5 + 0.2  # sqrt(half_extents) + turtlebot_diam + buffer
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

            # Box placement in room?
            if not self.is_point_in_room(box_initial_pos):
                continue

            # Box placement clear of other obsacles?
            if not self.check_avoid_poses(box_initial_pos, buffer=box_max/2):
                continue

            # # Avoid boxes on path
            # if self.is_close_to_path(box_initial_pos, min_distance_by_box_size):
            #     continue

            break

        return box_initial_pos

    def load_room(self, env):
        # STATIC
        # =============================================
        # filename = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
        # self.floor = p.loadURDF(filename)
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), "plane100.obj"))
        visual_id = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), "plane100.obj"))
        body_id = p.createMultiBody(
            basePosition=[0, 0, 0.01],
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id
        )
        self.floor = body_id
        self.env.scene.floor_body_ids += [self.floor]
        self.env.simulator.load_object_in_renderer(None, self.floor, SemanticClass.SCENE_OBJS, use_pbr=False, use_pbr_mapping=False)

        self.walls = self.load_walls_igibson(self.config_room)

        self.human_pos_indicator = VisualMarkerHRL(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 0.8, 0.5],
            radius=0.25,
            length=0.05,
            # initial_offset=[0, 0, 0.05 / 2.0],
        )
        self.human_pos_indicator.load(env.simulator)

        # RANDOMIZATION
        # =============================================
        for i in range(self.config_room["cube_large_number"]):
            max_dim = max(np.array(self.config_room["cube_large_dim"])[:2])
            pos = self.get_position_for_box(env, max_dim)
            orn = self.get_orientation_for_box()
            dim = self.config_room["cube_large_dim"]
            cube = Cube(pos, dim, mass=0, color=self.config_room["color_obstacles"])
            cube.load(env.simulator)
            cube.set_position(pos)
            cube.set_orientation(orn)
            self.cubes_large.append(cube)
            self.avoid_poses.append(pos)

            self.config_room["randomization"]["cubes_large"][i] = dict()
            self.config_room["randomization"]["cubes_large"][i]["pos"] = pos
            self.config_room["randomization"]["cubes_large"][i]["orn"] = orn

        for i in range(self.config_room["cube_small_number"]):
            max_dim = max(np.array(self.config_room["cube_small_dim"])[:2])
            pos = self.get_position_for_box(env, max_dim)
            orn = self.get_orientation_for_box()
            dim = self.config_room["cube_small_dim"]
            cube = Cube(pos, dim, mass=0, color=self.config_room["color_obstacles"])
            cube.load(env.simulator)
            cube.set_position(pos)
            cube.set_orientation(orn)
            self.cubes_small.append(cube)
            self.avoid_poses.append(pos)

            self.config_room["randomization"]["cubes_small"][i] = dict()
            self.config_room["randomization"]["cubes_small"][i]["pos"] = pos
            self.config_room["randomization"]["cubes_small"][i]["orn"] = orn

    def randomize_room(self, env):
        self.avoid_poses = list()
        # RANDOMIZATION
        # =============================================
        for i, cube in enumerate(self.cubes_large):
            max_dim = max(np.array(self.config_room["cube_large_dim"])[:2])
            pos = self.get_position_for_box(env, max_dim)
            orn = self.get_orientation_for_box()
            cube.unhide()
            cube.set_position(pos)
            cube.set_orientation(orn)
            self.avoid_poses.append(pos)

            self.config_room["randomization"]["cubes_large"][i]["pos"] = pos
            self.config_room["randomization"]["cubes_large"][i]["orn"] = orn

        for i, cube in enumerate(self.cubes_small):
            max_dim = max(np.array(self.config_room["cube_small_dim"])[:2])
            pos = self.get_position_for_box(env, max_dim)
            orn = self.get_orientation_for_box()
            cube.unhide()
            cube.set_position(pos)
            cube.set_orientation(orn)
            self.avoid_poses.append(pos)

            self.config_room["randomization"]["cubes_small"][i]["pos"] = pos
            self.config_room["randomization"]["cubes_small"][i]["orn"] = orn


    def load_walls_igibson(self, config):
        walls = list()
        walls.append(
            Cube(
                dim=[config["width"]/2, config["wall_thickness"]/2, config["height"]/2],
                pos=[0, +config["length"]/2, config["height"]/2],
                mass=0,
                color=config["color_wall"],
            )
        )
        walls.append(
            Cube(
                dim=[config["width"] / 2, config["wall_thickness"] / 2, config["height"]/2],
                pos=[0, -config["length"] / 2, config["height"] / 2],
                mass=0,
                color=config["color_wall"],
            )
        )
        walls.append(
            Cube(
                dim=[config["wall_thickness"] / 2, config["length"] / 2, config["height"]/2],
                pos=[+config["width"] / 2, 0, config["height"] / 2],
                mass=0,
                color=config["color_wall"],
            )
        )
        walls.append(
            Cube(
                dim=[config["wall_thickness"] / 2, config["length"] / 2, config["height"]/2],
                pos=[-config["width"] / 2, 0, config["height"] / 2],
                mass=0,
                color=config["color_wall"],
            )
        )
        for wall in walls:
            wall.load(self.env.simulator)
        return walls

    def hide_obstacles(self):
        for cube in self.cubes_small + self.cubes_large:
            cube.hide()
        self.env.simulator_step()

    def unhide_obstacles(self):
        for cube in self.cubes_small + self.cubes_large:
            cube.unhide()
        self.env.simulator_step()

    def is_point_in_room(self, point):
        if abs(point[0]) >= self.config_room["width"]/2 - self.config_room["wall_thickness"] / 2:
            return False
        if abs(point[1]) >= self.config_room["length"] / 2 - self.config_room["wall_thickness"] / 2:
            return False
        return True

    def get_orientation_for_box(self):
        yaw = np.random.choice([0, np.pi/2])
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        return orn