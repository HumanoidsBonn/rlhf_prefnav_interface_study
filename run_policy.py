from __future__ import absolute_import, division, print_function
import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import jorgeplot_utils as jg
import numpy as np
from stable_baselines3 import TD3
from torch import FloatTensor

from gibson2.TrajectoryPlotteriGibson import Trajectory
from gibson2.interface.UserInterfaceAnalysis import UserInterfaceAnalysis
from gibson2.interface.UserInterfaceAnalysisPDMORL import UserInterfaceAnalysisPDMORL
from gibson2.metrics.metrics_distance.frechet_distance import Frechet_distance
from pd_morl_jorge.demo_tools.trajectory_utils import create_demonstration_rounding_human
from pd_morl_jorge.lib.utilities.MORL_utils import generate_w_batch_test, generate_w_batch_test_baseline_objective

sys.path.append('../')
from pd_morl_jorge.learner import *
from pd_morl_jorge.child_process_funcs.igibson import *
from gibson2.utils import create_env_and_config

import logging
import os
import yaml

from igibson_rlhf.environments.Base_Env import BaseEnv
from igibson_rlhf.feature_extractors.basic import TaskObsAndScanExtractor
from igibson_rlhf.wrapper.observation_wrapper.min_pool_rays import MinPoolRays
from stable_baselines3 import TD3
import jorgeplot_utils as jg
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

# =================================================================================================================
# SETTINGS
# =================================================================================================================
yaml_file = "RLHFStudy.yml"

# =================================================================================================================
# EXECUTION
# =================================================================================================================
def main():
    global yaml_file
    train_path = os.path.join("/Users/jorgedeheuvel/Nextcloud/PhD_Uni_Bonn/Software/rlhf_gui/preference_vr_2d_user_study/core")

    # ARGPARSER
    # ===========================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=False, default=yaml_file, type=str)
    parser.add_argument("-v", "--verbose", required=False, default=True, type=bool)
    parser.add_argument("-c", "--continue", required=False, default=False, action='store_true')
    parser.add_argument("-id", "--continue_id", required=False, default='', type=str)
    # parser.add_argument("-a", "--actor_num", required=False, default='421491', type=str)
    parser.add_argument("-a", "--actor_num", required=False, default='final', type=str)
    parser.add_argument("--nowandb", required=False, default=True, action='store_true')
    parser.add_argument("--gui_pb", required=False, default=False, action='store_true')
    parser.add_argument("--gui_gibson", required=False, default=False, action='store_true')
    parser.add_argument("--render", required=False, default=False, action='store_true')
    parser.add_argument("--load_model", required=False, default=False, action='store_true')
    parser.add_argument("--train_path_experiment", required=False, default=os.getcwd(), type=str)
    args = vars(parser.parse_args())
    args["train_path"] = train_path
    print("ArgumentParser:", args)

    # PREPARATIONS ENV
    # ===========================================================
    config_rl = common_init_igibson(args)
    args_parse = args
    config_rl.update(args_parse)
    args = SimpleNamespace(**config_rl)

    # Initialize Env
    # ===========================================================
    env, _, config_igibson = create_env_and_config(
        args.yaml,
        train_path,
        export_observations=False,
        return_configs=True,
        gui_pb=args.gui_pb,
        wrap_env=False,
        update_env_params=False,
    )

    env = MinPoolRays(env)
    env.reset()

    # USER INTERFACE
    # ===========================================================
    ui = UserInterfaceAnalysis(env, config_rl["train_path_experiment"])
    ui.set_configs(config_rl, config_igibson.yaml_data)
    ui.plot_state_animation = False
    ui.plot_map_animation = False

    # LOAD ACTOR
    # ===========================================================
    tuned_policies = dict()

    policy_paths = {
        "VR": os.path.join(
            "/Users/jorgedeheuvel/test_dataset/training_rl/25_02_28__alignment_scratch_offset_mixed_enquery_stronger_vr_dutiful-frost-9/best_model/best_model.zip"),
        "2D-TD": os.path.join(
            "/Users/jorgedeheuvel/test_dataset/training_rl/25_02_28__alignment_scratch_offset_mixed_enquery_stronger_td_vague-wildflower-8/best_model/best_model.zip"),
        "2D-FPV": os.path.join(
            "/Users/jorgedeheuvel/test_dataset/training_rl/25_02_28__alignment_scratch_offset_mixed_enquery_stronger_fpv_dandy-water-7/best_model/best_model.zip"),
        "BL": os.path.join(
            "/Users/jorgedeheuvel/test_dataset/training_rl/25_03_01__alignment_bl_desert-bird-29/best_model/best_model.zip"),

    }

    for policy_label, path in policy_paths.items():
        tuned_policies[policy_label] = TD3.load(path, env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})

    # HELPER FUNCTIONS
    # ===========================================================
    def seed_and_reset(env, seed):
        env.seed(seed)
        env.reset()
        env.reset()


    # Init subplots
    # ========================================================================
    for p, (policy_label, policy) in enumerate(tuned_policies.items()):
        env.activate_evaluation_mode()
        env.reset()

        # Run simulation
        # ========================================================================
        state = env.get_state()
        while True:
            action, _ = policy.predict(state, deterministic=True)
            action =  np.reshape(action, (-1))
            state, reward, done, info = env.step(action)
            if done:
                break



if __name__ == "__main__":
    main()
    exit(0)
