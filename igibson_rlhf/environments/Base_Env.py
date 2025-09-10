import random
import copy
import os

import igibson
import numpy as np
import pybullet as p
from gym.spaces import Box
from collections import OrderedDict
from transforms3d.euler import euler2quat
from igibson.utils.utils import quatToXYZW
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson import object_states
from igibson.robots.robot_base import BaseRobot
from igibson.envs.igibson_env import iGibsonEnv


from igibson_rlhf.scenes.igibson_indoor_scene_custom import InteractiveIndoorSceneCustom
from igibson_rlhf.sensors.scan_sensor_rplidar import ScanSensorRplidar
from igibson_rlhf.tasks.rlhf_study_scene import RLHFStudyScene

import logging


from igibson.object_states import AABB
from igibson.object_states.utils import detect_closeness
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator_vr import SimulatorVR
from igibson.utils.utils import parse_config

from igibson_rlhf.utils import get_objects_in_scene, set_random_seed

from igibson_rlhf.scenes.simulator_custom import SimulatorCustom


class BaseEnv(iGibsonEnv):
    def __init__(
            self,
            config_file,
            normalize_observation=True,
            *args,
            **kwargs,
    ):
        self.config_file_backup = copy.deepcopy(parse_config(config_file))
        self.log = None
        self._init_logger()
        self.evaluation = False
        self.robot_config_backup = copy.deepcopy(config_file["robot"])
        super().__init__(config_file, *args, **kwargs)

        self.normalize_observation = normalize_observation
        self.observation_space_mean = None
        self.observation_space_delta = None

        self.task_randomization_freq = self.config.get("task_randomization_freq", 1)
        self.scene_randomization_freq = self.config.get("scene_randomization_freq", None)
        self.robot_randomization_freq = self.config.get("robot_randomization_freq", None)

        self.scene_id_list = self.config.get("scene_id_list", None)

        self.update_observation_space_callback = list()
        self.wrapper_list = list()
        # self.map_variables()

        self.demonstration_mode = False
        self.state = None
        self.action_last = np.zeros(2)
        self.print_collision_class = True

        self.render_screen = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.output_frames_per_second': int(1/self.simulator.render_timestep)
            }

        self.action_timestep_robot = self.config.get("action_timestep_robot", self.action_timestep)

        #testing
        # self.simulator.viewer = Viewer_custom(simulator=self.simulator, renderer=self.simulator.renderer)
        light_modulation_map_filename = os.path.join(
            igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
        )
        self.rendering_settings = MeshRendererSettings(
            enable_shadow=self.config.get("enable_shadow", False),
            enable_pbr=self.config.get("enable_pbr", False),
            msaa=self.config.get("msaa", False),
            texture_scale=self.config.get("enable_pbr", 1.0),
            optimized=self.config.get("optimized_renderer", True),
            load_textures=self.config.get("load_texture", True),
            hide_robot=self.config.get("hide_robot", True),
            light_modulation_map_filename=light_modulation_map_filename,
        )
        if self.mode in ["gui_interactive", "gui_non_interactive", "headless"]:
            self.simulator =  SimulatorCustom(mode=self.mode,
                                              physics_timestep=self.physics_timestep,
                                              action_timestep=self.action_timestep,
                                              image_width= self.config.get("image_width", 128),
                                              image_height=self.config.get("image_height", 128),
                                              vertical_fov=self.config.get("vertical_fov", 90),
                                              rendering_settings=self.rendering_settings,
                                              use_pb_gui=False)
            self.simulator.reload()
        self.load()

    def seed(self, seed=None):
        set_random_seed(seed)

    def land(self, obj, pos, orn, verbose=True):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if any(len(p.getContactPoints(bodyA=body_id)) > 0 for body_id in obj.get_body_ids()):
                land_success = True
                break

        if not land_success and verbose:
            log = logging.getLogger(__name__)
            log.warning("Object failed to land.")

        if is_robot:
            obj.reset()
            obj.keep_still()
        return land_success


    def set_action_timestep(self, timestep):
        """
        :return:  The previous action timestep
        """
        prev = self.simulator.render_timestep

        self.action_timestep = timestep
        self.simulator.render_timestep = timestep
        self.simulator.set_timestep(self.simulator.physics_timestep, timestep)
        self.simulator.physics_timestep_num = self.simulator.render_timestep / self.simulator.physics_timestep
        assert self.simulator.physics_timestep_num.is_integer(), "render_timestep must be a multiple of physics_timestep"
        self.simulator.physics_timestep_num = int(self.simulator.physics_timestep_num)

        return prev

    def activate_demo_mode(self):
        self.demonstration_mode = True

    def deactivate_demo_mode(self):
        self.demonstration_mode = False

    def activate_evaluation_mode(self):
        self.evaluation = True

    def deactivate_evaluation_mode(self):
        self.evaluation = False

    def _init_logger(self, debug=False):
        if debug:
            loglevel = logging.DEBUG
        else:
            loglevel = logging.INFO

        stream_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(loglevel)

        self.log = logging.Logger(name="POC")
        self.log.addHandler(stream_handler)

    def load_task_setup(self):
        """
        Load task setup.
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep**2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        # task
        if "task" not in self.config:
            self.task = DummyTask(self)
        elif self.config["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config["task"] == "interactive_nav_random":
            self.task = InteractiveNavRandomTask(self)
        elif self.config["task"] == "dynamic_nav_random":
            self.task = DynamicNavRandomTask(self)
        elif self.config["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config["task"] == "room_rearrangement":
            self.task = RoomRearrangementTask(self)

        # ==============================================================================
        # Our own tasks go here:
        # ==============================================================================
        elif self.config["task"] == "rlhf_study_scene":
            self.task = RLHFStudyScene(self)
        else:
            try:
                import bddl

                with open(os.path.join(os.path.dirname(bddl.__file__), "activity_manifest.txt")) as f:
                    all_activities = [line.strip() for line in f.readlines()]

                if self.config["task"] in all_activities:
                    self.task = BehaviorTask(self)
                else:
                    raise Exception("Invalid task: {}".format(self.config["task"]))
            except ImportError:
                raise Exception("bddl is not available.")
        self.task.reset(self)


    def load_observation_space(self):
        """
        Load observation space with custom sensors.
        """
        # Call igibson specific implementation for proprietary sensors.
        super().load_observation_space()
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        # Take over sensors from super() call.
        sensors = self.sensors

        if "scan_rplidar" in self.output:
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan_rplidar")
        if "occupancy_grid" in self.output:
            # Fixes dependency of ScanSensorRplidar on occupancy_grid in scan_modalities from super()
            scan_modalities.append("occupancy_grid")
        if "task_obs" in self.output and hasattr(self.task, "task_observation_range"):
            low = np.concatenate(self.task.task_observation_range[0])
            high = np.concatenate(self.task.task_observation_range[1])
            observation_space["task_obs"] = Box(low, high)
        if "reward_weight" in self.output:
            reward_weight_len = len(self.task.reward_controller.reward_functions_scalable) \
                                + len(self.task.reward_controller.reward_functions_controllable)
            low = np.zeros(reward_weight_len)
            high = np.ones(reward_weight_len)
            observation_space["reward_weight"] = Box(low, high)


        if len(scan_modalities) > 0:
            if "scan_rplidar" in scan_modalities:
                sensors["scan_occ"] = ScanSensorRplidar(self, scan_modalities)

        if "scan_rear" in scan_modalities:
            sensors["scan_occ_rear"] = ScanSensorRplidar(self, scan_modalities, rear=True)

        for key in observation_space:
            self.observation_space.spaces[key] = observation_space[key]

        self.sensors = sensors

    def reset(self):
        """
        Reset episode.
        """
        # Robot randomization
        # =========================================================================
        if self.robot_randomization_freq is not None and self.current_episode > 0 and not self.evaluation:
            if self.current_episode % self.robot_randomization_freq == 0:
                self.robot_randomization()

        # Scene randomization
        # =========================================================================
        if self.scene_randomization_freq is not None and self.current_episode > 0 and not self.evaluation:
            if self.current_episode % self.scene_randomization_freq == 0:
                scene_id = self.sample_scene()
                self.change_scene(scene_id)

        self.randomize_domain()
        # Move robot away from the scene.
        self.robots[0].set_position([100.0, 100.0, 100.0])

        # Task randomization
        # =========================================================================
        if self.task_randomization_freq is not None and not self.evaluation:
            if self.current_episode % self.task_randomization_freq == 0:
                self.task.reset(self)
            else:
                self.task.restore(self)
        else:
            self.task.reset(self)

        self.simulator.sync(force_sync=True)
        state = self.get_state()
        self.reset_variables()

        return state

    def change_scene(self, scene_id, scene_type=None, update_obs_space=True):
        # TODO causes observation space to change
        if self.config["scene_id"] == scene_id:
            self.log.info("Scene {} already loaded.".format(scene_id))
            return False
        self.log.info("Loading Scene {}...".format(scene_id))

        config_new = copy.deepcopy(self.config_file_backup)
        config_new["scene_id"] = scene_id
        config_new["scene"] = scene_type if scene_type is not None else config_new["scene"]

        # Special case fix: Pomaria_2_int has only a bedroom
        if config_new["scene_id"] == "Pomaria_2_int":
            config_new["load_room_types"].append("bedroom")

        self.reload(config_new)

        if isinstance(self.simulator, SimulatorVR):
            self.simulator.main_vr_robot = None

        self.task.save_id = None

        if update_obs_space:
            # print("updating obs space after changed scene")
            self.update_observation_space()
        # shape_info = ["{} {}".format(key, self.observation_space[key].shape) for key in self.observation_space.keys()]
        # print("After Change_Scene: observation space  {}".format(shape_info))
        # if hasattr(self.task, "ui"):
        #     self.task.ui.update_record_params()
        self.log.info("Scene {} loaded into environment.".format(scene_id))
        return True



    def sample_scene(self):
        assert isinstance(self.scene_id_list, list), "No scene id list provided for scene randomization!"
        return random.choice(self.scene_id_list)


    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        self.task.step(self)
        if action is not None:
            self.robots[0].apply_action(action)
            self.action_last = action
        collision_links = self.run_simulation()

        # TODO Why necessary when action_timestep != 1/5 s?:
        collision_links = self.filter_floor_collision(collision_links)
        if self.print_collision_class and self.evaluation:
            self.get_collision_object_type_efficient(collision_links)

        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        self.state = self.get_state()
        info = {}
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)

        self.populate_info(info)
        if done and self.automatic_reset:
            info["last_observation"] = self.state
            self.state = self.reset()

        return self.state, reward, done, info

    def get_collision_object_type(self, collision_links):
        for link in collision_links:
            id_object = link[2]
            for obj in self.scene.get_objects():
                if id_object in obj.get_body_ids():
                    str = get_objects_in_scene([obj.class_id])
                    print("Collision with class", obj.class_id, str)

    def get_collision_object_type_efficient(self, collision_links):
        if not "scene_id" in self.config.keys():
            return
        if self.config["scene_id"] == "empty":
            return
        for link in collision_links:
            id_object = link[2]
            collision_obj = self.scene.objects_by_id[id_object]
            str = get_objects_in_scene([collision_obj.class_id])
            self.log.info("Collision with class {}, id {}".format(str, collision_obj.class_id))

    def filter_floor_collision(self, collision_links):
        if hasattr(self.task, "dedicated_collision_objects"):
            collision_links_filtered = list()
            for link in collision_links:
                id_object = link[2]
                if id_object in self.task.dedicated_collision_objects:
                    collision_links_filtered.append(link)

            return collision_links_filtered

        if hasattr(self.scene, "objects_by_category"):
            collision_links_filtered = list()
            for link in collision_links:
                id_object = link[2]
                floor_ids = self.scene.objects_by_category["floors"][0].get_body_ids()
                if not id_object in floor_ids:
                    collision_links_filtered.append(link)
                # else:
                #     print("Floor filtered.")
            return collision_links_filtered

        if hasattr(self.scene, "floor_body_ids"):
            collision_links_filtered = list()
            for link in collision_links:
                id_object = link[2]
                if not id_object in self.scene.floor_body_ids:
                    collision_links_filtered.append(link)
                # else:
                #     print("Floor filtered.")
            return collision_links_filtered

        return collision_links

    def get_state_full(self):
        return super().get_state()

    def compute_observation(self, action):
        return self.get_state()

    def get_robot_pose_and_yaw(self):
        pos = self.robots[0].get_position()
        orn = self.robots[0].get_orientation()
        return pos[0], pos[1], orn[2]

    def update_observation_space(self, task_range=3):
        if self.update_observation_space_callback:
            for fn in self.update_observation_space_callback:
                self.log.info("Executing update_observation_space_callback func {}".format(fn))
                fn()

    def test_valid_position_on_trav_map(self, pos):
        """Looks up a point and checks whether it is traversible on map"""
        if not hasattr(self.scene, "world_to_map"):
            return True
        floor_ind = self.scene.world_to_map(np.array(pos[:2]))
        trav_map = self.scene.floor_map[0]

        # trav_map = np.flip(trav_map, axis=0)
        # print(floor_ind)
        # print(trav_map.shape)
        # print(trav_map[floor_ind[0], floor_ind[1]])
        # x_samples = np.random.uniform(0, trav_map.shape[0], 500).astype("int")
        # y_samples = np.random.uniform(0, trav_map.shape[1], 500).astype("int")
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(trav_map)
        # for i in range(len(x_samples)):
        #     print()
        #     if trav_map[x_samples[i], y_samples[i]] == 0:
        #         plt.plot(y_samples[i],  x_samples[i], "ro")
        #     else:
        #         plt.plot(y_samples[i], x_samples[i], "go")
        # plt.plot(floor_ind[1], floor_ind[0], "bo")
        # plt.show()

        if trav_map[floor_ind[0], floor_ind[1]] == 0:
            # Non traversible!
            return False
        else:
            return True

    def test_valid_position(self, obj, pos, orn=None, ignore_self_collision=False):
        """
        Test if the robot or the object can be placed with no collision.
        Custom version, that also deals with objects which are not on trav_map.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param ignore_self_collision: whether the object's self-collisions should be ignored.
        :return: whether the position is valid
        """
        if not self.test_valid_position_on_trav_map(pos):
            # self.log.info("test_valid_position_on_trav_map failed")
            return False

        # # Check for object collisions in radius that is similar to trav_map erosion
        # # ===========================================================================
        # TODO This is too slow. It calls a physics step every time, which is stupid.
        # check_radius = self.scene.trav_map_erosion * self.scene.trav_map_resolution / 2
        # circular_checks_n = 6
        # offset_angle = np.arange(6) * 2 * np.pi / circular_checks_n
        # dx = np.cos(offset_angle) * check_radius
        # dy = np.sin(offset_angle) * check_radius
        # dz = np.zeros_like(offset_angle)
        # offset = np.stack([dx, dy, dz], axis=1)
        # pos_check = offset + pos
        # for i in range(circular_checks_n):
        #     if not super().test_valid_position(obj, pos_check[i, :], orn, ignore_self_collision):
        #         return False
        # return True

        if not super().test_valid_position(obj, pos, orn, ignore_self_collision):
            # self.log.info("test_valid_position failed")
            return False
        return True

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        # first set the correct orientation
        obj.set_position(pos)
        obj.set_orientation(quatToXYZW(euler2quat(*orn), "wxyz"))
        # get the AABB in this orientation
        lower, _ = obj.states[object_states.AABB].get_value()
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def robot_randomization(self):

        # Robot camera randomization
        # =====================================================================
        if not hasattr(self.robots[0].eyes, "set_position_orientation_delta"):
            self.log.warning("'RobotLink' object has no attribute 'set_position_orientation_delta'")
            return

        pos_noise_range = self.config.get("robot_camera_randomization_pos", 0)
        orn_noise_range = self.config.get("robot_camera_randomization_orn", 0)

        pos_noise = np.random.uniform(-pos_noise_range, +pos_noise_range, (3,))
        orn_noise = np.random.uniform(-orn_noise_range, +orn_noise_range, (4,))

        self.robots[0].eyes.set_position_orientation_delta(pos_noise, orn_noise)
        self.log.info("Robot camera randomization delta to {} (pos), {} (orn).".format(pos_noise, orn_noise))
        self.log.info("Robot camera pose {} (pos), {} (orn).".format(*self.robots[0].eyes.get_position_orientation()))

        # raise NotImplementedError("Robot randomization not set for this environment.")

    @property
    def _max_episode_steps(self):
        return self.config.get("max_step", 150)

    @_max_episode_steps.setter  # the property decorates with `.setter` now
    def _max_episode_steps(self, value):  # name, e.g. "attribute", is the same
        self.config.set("max_step", value)

    def map_variables(self):
        from_to = {"_max_episode_steps": "max_step"}
        for key in from_to:
            def get():
                return getattr(self, from_to[key])
            def set(x):
                setattr(self, from_to[key], x)

            setattr(self, str(key), property(fget=get, fset=set))

    def load(self):
        """
        Custom version of the env_base.py load function, in order to load custom scene Object.
        """
        if self.config["scene"] == "empty":
            scene = EmptyScene(floor_plane_rgba=[0.7, 0.7, 0.7, 1.0])
        elif self.config["scene"] == "stadium":
            scene = StadiumScene()
        elif self.config["scene"] == "gibson":
            scene = StaticIndoorScene(
                self.config["scene_id"],
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                pybullet_load_texture=self.config.get("pybullet_load_texture", False),
            )
        elif self.config["scene"] == "igibson":
            urdf_file = self.config.get("urdf_file", None)
            if urdf_file is None and not self.config.get("online_sampling", True):
                urdf_file = "{}_task_{}_{}_{}_fixed_furniture".format(
                    self.config["scene_id"],
                    self.config["task"],
                    self.config["task_id"],
                    self.config["instance_id"],
                )
            include_robots = self.config.get("include_robots", True)
            scene = InteractiveIndoorSceneCustom(
                self.config["scene_id"],
                urdf_file=urdf_file,
                waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
                num_waypoints=self.config.get("num_waypoints", 10),
                build_graph=self.config.get("build_graph", False),
                trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
                trav_map_erosion=self.config.get("trav_map_erosion", 2),
                trav_map_type=self.config.get("trav_map_type", "with_obj"),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get("should_open_all_doors", False),
                load_object_categories=self.config.get("load_object_categories", None),
                not_load_object_categories=self.config.get("not_load_object_categories", None),
                load_room_types=self.config.get("load_room_types", None),
                load_room_instances=self.config.get("load_room_instances", None),
                merge_fixed_links=self.config.get("merge_fixed_links", True)
                and not self.config.get("online_sampling", False),
                include_robots=include_robots,
            )
            # TODO: Unify the function import_scene and take out of the if-else clauses.
            first_n = self.config.get("_set_first_n_objects", -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)

        self.simulator.import_scene(scene)

        # Get robot config
        # robot_config = self.config["robot"]
        robot_config = copy.deepcopy(self.robot_config_backup)
        print("robot config ::::: ", robot_config )
        # If no robot has been imported from the scene
        if len(scene.robots) == 0:
            # Get corresponding robot class
            robot_name = robot_config.pop("name")
            assert robot_name in REGISTERED_ROBOTS, "Got invalid robot to instantiate: {}".format(robot_name)
            robot = REGISTERED_ROBOTS[robot_name](**robot_config)

            self.simulator.import_object(robot)

            # The scene might contain cached agent pose
            # By default, we load the agent pose that matches the robot name (e.g. Fetch, BehaviorRobot)
            # The user can also specify "agent_pose" in the config file to use the cached agent pose for any robot
            # For example, the user can load a BehaviorRobot and place it at Fetch's agent pose
            agent_pose_name = self.config.get("agent_pose", robot_name)
            if isinstance(scene, InteractiveIndoorSceneCustom) and agent_pose_name in scene.agent_poses:
                pos, orn = scene.agent_poses[agent_pose_name]

                if agent_pose_name != robot_name:
                    # Need to change the z-pos - assume we always want to place the robot bottom at z = 0
                    lower, _ = robot.states[AABB].get_value()
                    pos[2] = -lower[2]

                robot.set_position_orientation(pos, orn)

                if any(
                    detect_closeness(
                        bid, exclude_bodyB=scene.objects_by_category["floors"][0].get_body_ids(), distance=0.01
                    )
                    for bid in robot.get_body_ids()
                ):
                    self.log.warning("Robot's cached initial pose has collisions.")

        self.scene = scene
        self.robots = scene.robots

        # From igibson_env.py
        self.load_action_space()
        self.load_task_setup()
        self.load_observation_space()
        self.load_miscellaneous_variables()

    def init_live_plotter(self):
        self.log.warning("init_live_plotter() not implemented in this env")
        pass

    def render(self, mode="human"):
        from igibson.render.viewer import ViewerSimple
        import cv2

        if mode == "human":
            if self.render_screen is None:
                self.render_screen = ViewerSimple(renderer=self.simulator.renderer)
            self.render_screen.update()

        elif mode == "rgb_array":
            frames = self.simulator.renderer.render_robot_cameras(modes=("rgb"), cache=False)
            if len(frames) > 0:
                frame = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
                frame = 255 * frame  # Now scale by 255
                frame = frame.astype(np.uint8)
                return frame
        else:
            raise RuntimeError("Render mode {} unknown!".format(mode))

    def get_state(self):
        """
        Get the current observation.

        :return: observation as a dictionary
        """
        state = OrderedDict()
        if "task_obs" in self.output:
            task_obs = self.task.get_task_obs(self)
            # if hasattr(self.task, "task_observation_range"):
            #     delta = abs(self.observation_space["task_obs"].high - self.observation_space["task_obs"].low)
            #     mean = (self.observation_space["task_obs"].high + self.observation_space["task_obs"].low) / 2
            #     task_obs = (task_obs - mean) / delta
            state["task_obs"] = task_obs
        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "scan_occ_rear" in self.sensors:
            scan_obs = self.sensors["scan_occ_rear"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)
        if "proprioception" in self.output:
            state["proprioception"] = np.array(self.robots[0].get_proprioception())
        if "reward_weight" in self.output:
            if hasattr(self.task, "reward_controller"):
                state["reward_weight"] = np.array(self.task.reward_controller.last_reward_weight)
            else:
                raise ValueError("Task has no reward controller to get reward_weight from")

        return state


