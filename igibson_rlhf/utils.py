import random
import numpy as np
import os
import datetime
import matplotlib
import cv2
import torch

import igibson_rlhf.Config as g2c
import igibson_rlhf.environments.utils.env_utils as g2eu
# from trajectory_demo_proof_of_concept.networks.utils.network_selector import get_network_class
from igibson.utils.utils import l2_distance
from sb3_extensions.common.init_utils import wrap_environment

# Store the mapping from class labels to colors for segmentation

MAX_UNIQUE_COLORS = 20
seg_mask_labels_to_colors = {}

####################################
# Printing class segmentation data #
####################################
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID

UNIQUE_OBJECT_IDENTIFIERS = {0: "robot_0", 1: "robot_1", 2: "robot_2", 3: "robot_3", 398: "PERSON"}


def get_class_id_to_class_name():
    """
    See:  https://github.com/StanfordVL/iGibson/blob/master/igibson/utils/semantics_utils.py
    """
    CLASS_ID_TO_CLASS_NAME = UNIQUE_OBJECT_IDENTIFIERS
    for key in CLASS_NAME_TO_CLASS_ID:
        if key == 'PERSON':
            print("PERSON found in CLASS_NAME_TO_CLASS_ID and ID: ", CLASS_NAME_TO_CLASS_ID[key])
        # if key == "agent":
        #     continue # Special case
        CLASS_ID_TO_CLASS_NAME.update({CLASS_NAME_TO_CLASS_ID[key]: key})

    return CLASS_ID_TO_CLASS_NAME


CLASS_ID_TO_CLASS_NAME = get_class_id_to_class_name()


def get_objects_in_scene(seg_mask):
    """
    :returns: a list of the objects in the semantic segmentaion of the scene
    """
    global CLASS_ID_TO_CLASS_NAME

    unq_objs = np.unique(seg_mask)
    ret = []
    for obj in unq_objs:
        try:
            ret.append(str(CLASS_ID_TO_CLASS_NAME[obj]))
        except:
            ret.append("unknown")

    return ret

    # Legacy:  Print string
    # unq_objs = np.unique(seg_mask)
    # print_string = "Objects in scene:\n"
    # for obj in unq_objs:
    #     print_string += "\t" + str(CLASS_ID_TO_CLASS_NAME[obj])

    # print(print_string)


def make_folder(path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def get_timestamp():
    return str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")


def get_task_obj_str(state):
    """
    :return: the string for printing the task observation
    """
    ret = ""
    if "task_obs" in state.keys():
        # Difference in x/y from the robot to the goal position.
        # The origin is centered at the robot.
        #       - See igibson.tasks.PointNavFixedTask.get_task_obs(...)
        diff_x = state["task_obs"][0]
        diff_y = state["task_obs"][1]
        distance_to_goal = l2_distance([0., 0.], [diff_x, diff_y])

        ret = f"Dist To Goal:  {round(distance_to_goal, 2)} m.  \t"
        ret += f"Robot Velocity:  {round(state['task_obs'][2], 3)} m/s    \t"
        ret += f"Robot Angular velocity:  {round(state['task_obs'][3], 2)} rad"
        # print(state["task_obs"])

    return ret


def get_task_obj_str_single_ped_nav(state):
    """
    :return: the string for printing the task observation
    """
    ret = ""
    if "task_obs" in state.keys():
        ret = f"Dist To Goal:  {round(state['task_obs'][0], 2)} m  \t"
        ret += f"Angle To Goal:  {round(state['task_obs'][1], 2)} rad \t"
        ret += f"Human in FOV:  {state['task_obs'][2]} \t"
        ret += f"Dist To Human:  {round(state['task_obs'][3], 2)} m \t"
        ret += f"Angle To Human:  {round(state['task_obs'][4], 2)} rad \t"
    return ret


def show_rgb(state):
    """
    Show an RGB image if it is present in the state modalities
    """
    if "rgb" in state.keys():
        cv2.imshow('RGB Frame', state["rgb"][..., ::-1])
        cv2.waitKey(1)


def show_depth(state):
    """
    Show an RGB image if it is present in the state modalities
    """
    if "depth" in state.keys():
        # print("Depth SHape:  ",  state["depth"].shape)
        # print("Min Depth:  ",  state["depth"].min())
        # print("Max Depth:  ",  state["depth"].max())
        cv2.imshow('Depth Frame', state["depth"])
        cv2.waitKey(1)


def show_occ_map(state):
    """
    Display the occupance map for the current state
    """
    if "occupancy_grid" in state.keys():
        # print(type(state["occupancy_grid"]))
        # print(len(state["occupancy_grid"]))
        # print(state["occupancy_grid"].shape)
        # print("occupancy_grid type:  ",  type(state["occupancy_grid"]))
        # print("occupancy_grid dtype:  ",  state["occupancy_grid"].dtype)
        # print("occupancy_grid shape:  ",  state["occupancy_grid"].shape)
        # print("Min occupancy_grid:  ",  state["occupancy_grid"].min())
        # print("Max occupancy_grid:  ",  state["occupancy_grid"].max())
        cv2.imshow('Occ Map', state["occupancy_grid"])
        cv2.waitKey(1)




def create_seg_rgb(seg_mask):
    """
    :param seg_mask:  Np array of integers for each unique object in an image
    :return:  an RGB image with distinct colors for each object
    """
    # Get the number of unquie objects in the image
    unq_objs = np.unique(seg_mask)
    colors = matplotlib.cm.tab20(range(MAX_UNIQUE_COLORS))
    rgb_im = np.empty((seg_mask.shape[:2]) + (3,))
    prev_seg_col_map_length = len(seg_mask_labels_to_colors)

    for color_idx, obj in enumerate(unq_objs):
        # Get the next color
        # color_idx = (color_idx + prev_seg_col_map_length) % MAX_UNIQUE_COLORS
        color_idx = obj % MAX_UNIQUE_COLORS

        if obj in seg_mask_labels_to_colors.keys():
            color = seg_mask_labels_to_colors[obj]
        else:
            color = colors[color_idx][:3]
            seg_mask_labels_to_colors[obj] = color

        rgb_im[seg_mask.squeeze() == obj, :] = color

    return rgb_im


def set_random_seed(random_seed, set_torch=True):
    """
    Generalizing random seed
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if set_torch:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            # Important for reproducability
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def build_actor(config, config_igibson, env):
    # NETWORKS
    # ===========================================================
    if "input_shape" in config_igibson.keys():
        input_shape_actor = config_igibson["input_shape"]
    else:
        input_shape_actor = env.observation_space.shape

    net_kwargs = dict()
    if "state_preprocessor_config" in config_igibson.keys():
        net_kwargs["preprocessor_config"] = config_igibson["state_preprocessor_config"]
    if "model_config" in config.keys():
        net_kwargs["model_config"] = config["model_config"]

    actor_cls = get_network_class(config["actor_net"])
    actor = actor_cls(
        env.action_space.shape[0],
        input_shape_actor,
        batch_size=config["batch_size_rl"],
        **net_kwargs
    )
    # actor.compile(run_eagerly=True)
    return actor


def build_critic(config, config_igibson, env):
    if "input_shape" in config_igibson.keys():
        input_shape_critic = config_igibson["input_shape"]
    else:
        input_shape_critic = (env.observation_space.shape[0] + env.action_space.shape[0])

    net_kwargs = dict()
    if "state_preprocessor_config" in config_igibson.keys():
        net_kwargs["preprocessor_config"] = config_igibson["state_preprocessor_config"]
    if "model_config" in config.keys():
        net_kwargs["model_config"] = config["model_config"]

    critic_cls = get_network_class(config["critic_net"])
    try:
        critic = critic_cls(
            input_shape=input_shape_critic,
            batch_size=config["batch_size_rl"],
            action_dim=env.action_space.shape[0],
            **net_kwargs
        )
    except:
        critic = critic_cls(
            input_shape=input_shape_critic,
            batch_size=config["batch_size_rl"],
        )
    # critic.compile(run_eagerly=True)
    return critic


def create_config(yaml_file, train_path):
    # LOAD EXPERIMENT CONFIG FROM YAML FILE
    # ===========================================================
    yaml_path = os.path.join(train_path, '__experiment_configs__', yaml_file)
    with open(yaml_path, 'r') as stream:
        import yaml
        config_rl = yaml.safe_load(stream)
    print("#" * 60 + "\nEXPERIMENT CONFIG: \n" + "#" * 60)
    print(config_rl)
    return config_rl


def create_env_and_config(
        yaml_file,
        train_path,
        export_observations=False,
        return_configs=False,
        gui_pb=False,
        gui_gibson=False,
        load_scene_without_objects=False,
        goal_thresh=None,
        seed=12,
        wrap_env=False,
        update_env_params=True,
):
    # LOAD EXPERIMENT CONFIG FROM YAML FILE
    # ===========================================================
    config_rl = create_config(yaml_file, train_path)

    # =================================================================================================================
    # PREPARATIONS ENV
    # =================================================================================================================
    config_igibson = g2c.Config()
    config_igibson["yaml"] = os.path.join(train_path, config_rl['config_gibson'])
    if gui_gibson:
        config_igibson["gui"] = True
    if gui_pb:
        config_igibson["pb"] = True
    config_igibson.setup()
    if export_observations:
        if "rgb" not in config_igibson.yaml_data["output"]:
            config_igibson.yaml_data["output"].append("rgb")
        if "seg" not in config_igibson.yaml_data["output"]:
            config_igibson.yaml_data["output"].append("seg")
        if "depth" not in config_igibson.yaml_data["output"]:
            config_igibson.yaml_data["output"].append("depth")
        config_igibson.yaml_data["load_texture"] = True

    if load_scene_without_objects:
        config_igibson.yaml_data["load_room_types"] = list()
    config_igibson.yaml_data["dist_tol"] = goal_thresh if goal_thresh is not None else config_igibson.yaml_data[
        "dist_tol"]

    set_random_seed(seed)

    env = g2eu.load_env(config_rl['env_type'], config_igibson, config_rl)
    if update_env_params:
        from trajectory_demo_proof_of_concept.utils.utils import change_object_parameter
        change_object_parameter(env, config_rl)

    if hasattr(env, "update_observation_space"):
        env.update_observation_space()

    env.reset()

    if wrap_env:
        env = wrap_environment(env, config_rl, globals())

    if return_configs:
        return env, config_rl, config_igibson
    else:
        return env


def get_date_time_str():
    import datetime
    return "{date:%Y_%m_%d__%H_%M_%S}".format(date=datetime.datetime.now())