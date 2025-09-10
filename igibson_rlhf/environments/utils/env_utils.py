import sys
from igibson_rlhf.Config import *

# =================================================================================================================
# ENVIRONMENTS
# =================================================================================================================
def load_env_base(config_igibson):
    from igibson_rlhf.environments.Base_Env import BaseEnv
    env = BaseEnv(
        config_file=config_igibson.yaml_data,
        action_timestep=config_igibson["action_timestep"],
        physics_timestep=config_igibson["physics_timestep"],
        mode="headless" if not config_igibson["gui"] else "gui_interactive",
        use_pb_gui=config_igibson["pb"],
    )
    return env

def load_env_base_igibson(config_igibson):
    from igibson.envs.igibson_env import iGibsonEnv
    env = iGibsonEnv(
        config_file=config_igibson.yaml_data,
        action_timestep=config_igibson["action_timestep"],
        physics_timestep=config_igibson["physics_timestep"],
        mode="headless" if not config_igibson["gui"] else "gui_interactive",
        use_pb_gui=config_igibson["pb"],
    )
    return env


def load_env(type, config_igibson, config):
    if type == "base":
        env = load_env_base(config_igibson)
    elif type == "igibson_base":
        env = load_env_base_igibson(config_igibson)
    else:
        raise RuntimeError("Env type not recognized!")

    env.simulator.main_vr_robot = None
    return env
