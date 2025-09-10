from igibson_rlhf.utils import create_env_and_config
from igibson_rlhf.wrapper import *
from igibson_rlhf.Config import *
from igibson_rlhf.wrapper.reward_wrapper.human_reward_wrapper import HumanRewardModelWrapper

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.td3 import TD3

from sb3_extensions.callbacks.igibson_logger_callback import iGibsonLoggerCallback
from sb3_extensions.callbacks.evaluation_igibson import EvalCallbackIGibson
from sb3_extensions.common.init_utils import init_sb3_off_policy_model


# =================================================================================================================
# SETTINGS
# =================================================================================================================
yaml_file = "25_02_28__alignment_vr.yml"
# yaml_file = "25_03_01__alignment_bl.yml"

reward_paths = {"VR":"./resources/reward_models/VR-rewardmodel",
                    "2D_TD":"./resources/reward_models/2D_TD-rewardmodel",
                    "2D_FPV":"./resources/reward_models/2D_FPV-rewardmodel"}

# =================================================================================================================
# EXECUTION
# =================================================================================================================
def main():
    global train_path, model_epoch, mode, experiment_id, yaml_file, reward_paths
    train_path = os.path.join(os.getcwd())

    # ARGPARSER
    # ===========================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=False, default=yaml_file, type=str)
    parser.add_argument("-v", "--verbose", required=False, default=True, type=bool)
    parser.add_argument("-c", "--continue", required=False, default=False, action='store_true')
    parser.add_argument("-id", "--continue_id", required=False, default='', type=str)
    parser.add_argument("--nowandb", required=False, default=False, action='store_true')
    parser.add_argument("--gui_pb", required=False, default=False, action='store_true')
    parser.add_argument("--gui_gibson", required=False, default=False, action='store_true')
    parser.add_argument("--render", required=False, default=False, action='store_true')
    args = vars(parser.parse_args())
    print("ArgumentParser:", args)

    # PREPARATIONS ENV
    # ===========================================================
    print("CONFIG_RL: {}".format(args["yaml"]))
    env, config_rl, config_igibson = create_env_and_config(
        args["yaml"],
        train_path,
        export_observations=False,
        return_configs=True,
        gui_pb=args["gui_pb"],
    )

    # OUTPUT FOLDER
    # ===========================================================
    experiment_id = "{}".format(args["yaml"][:-4])
    train_path_experiment = os.path.join(
        os.getenv("HOME"),
        "test_dataset",
        "training_rl",
        experiment_id,
    )
    best_model_save_path = os.path.join(train_path_experiment, "best_model", "best_model.zip")

    # =================================================================================================================
    # WRAPPERS & REWARD MODEL
    # =================================================================================================================
    condition = config_rl["condition"]

    env = MinPoolRays(env)
    if condition != "BL":
        env = HumanRewardModelWrapper(
            env,
            reward_model_path=reward_paths[condition],
            concat=False,
            reward_scale=config_rl["reward_scale"],
            reward_offset=config_rl["reward_offset"],
            reward_scale_old=config_rl["reward_scale_old"],
            reward_model_balance=config_rl["reward_model_balance"],
        )
    env.reset()

    # =================================================================================================================
    # MODEL
    # =================================================================================================================
    model = init_sb3_off_policy_model(env, config_rl, globals())

    # =================================================================================================================
    # CALLBACKS
    # =================================================================================================================
    checkpoint_callback = CheckpointCallback(
        save_freq=config_rl["save_chkpt_freq_steps"],
        save_path=os.path.join(train_path_experiment, "checkpoints"),
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallbackIGibson(
        env,
        best_model_save_path=os.path.dirname(best_model_save_path),
        eval_freq=max(config_rl["eval_freq_steps"], config_rl["learning_starts"]),
        n_eval_episodes=config_rl["n_eval_episodes"],
        # callback_after_eval=stop_train_callback,
        record_eval=False,
        video_folder=os.path.join(train_path_experiment, "videos"),
        verbose=1 if args["verbose"] else 0,
    )
    logger_custom = iGibsonLoggerCallback()
    callback = list([eval_callback, logger_custom, checkpoint_callback,])
    callback = CallbackList(callback)

    # =================================================================================================================
    # LOGGER
    # =================================================================================================================
    log_path = os.path.join(train_path_experiment, "log")
    new_logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(new_logger)

    # =================================================================================================================
    # TRAIN
    # =================================================================================================================
    model.learn(
        total_timesteps=config_rl["total_timesteps"],
        log_interval=config_rl["log_interval"],
        callback=callback,
        progress_bar=True if args["verbose"] else False,
        reset_num_timesteps=False,
    )
    model.save(os.path.join(train_path_experiment, "final_model.zip"))

if __name__ == "__main__":
    main()
