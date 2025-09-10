import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

def init_sb3_off_policy_model(env, config, skript_globals, verbose=1):
    assert "agent" in config.keys(), "Agent type not defined in config."
    # AGENT KWARGS
    # ==========================================================================================================
    common_kwargs_off_policy = dict(
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        train_freq=config["train_freq"],
        tau=config["tau"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        # **config["common_kwargs_off_policy"] if "common_kwargs_off_policy" in config.keys() else dict()
    )

    agent_kwargs = dict()
    if "agent_kwargs" in config.keys():
        agent_kwargs = dict(config["agent_kwargs"])

    if "action_noise" in config.keys():
        noise_model = NormalActionNoise(
            mean=np.array(config["action_noise"]["mean"]),
            sigma=np.array(config["action_noise"]["sigma"]),
        )
        common_kwargs_off_policy["action_noise"] = noise_model

    if "replay_buffer" in config.keys():
        common_kwargs_off_policy["replay_buffer_class"] = skript_globals[config["replay_buffer"]["replay_buffer_class"]]
        common_kwargs_off_policy["replay_buffer_kwargs"] = config["replay_buffer"]["replay_buffer_kwargs"]

    # POLICY KWARGS
    # ==========================================================================================================
    policy_kwargs = dict()
    if "feature_extractor" in config.keys():
        print("Using Feature extractor {}!".format(config["feature_extractor"]["features_extractor_class"]))
        policy_kwargs["features_extractor_class"] = skript_globals[config["feature_extractor"]["features_extractor_class"]]
        policy_kwargs["features_extractor_kwargs"] = config["feature_extractor"]["features_extractor_kwargs"]

    if "net_arch" in config.keys():
        policy_kwargs["net_arch"] = config["net_arch"]

    if "share_features_extractor" in config.keys():
        policy_kwargs["share_features_extractor"] = config["share_features_extractor"]

    # CHOICE OF AGENT
    # ==========================================================================================================
    agent_cls = skript_globals[config["agent"]]
    model = agent_cls(
        config["policy"],
        env,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        **common_kwargs_off_policy,
        **agent_kwargs,
    )
    return model

def load_sb3_off_policy_model(env, config, path, skript_globals):
    assert "agent" in config.keys(), "Agent type not defined in config."
    # CHOICE OF AGENT
    # ==========================================================================================================
    print("Restoring model from checkpoint {}".format(path))
    agent_cls = skript_globals[config["agent"]]
    model = agent_cls.load(path, env, reset_num_timesteps=False)
    return model


def wrap_environment(env, config, skript_globals):
    """
    Function wraps gym environment in wrappers, that are defined in the config file and applies dedicated kwargs.
    """
    if not "wrapper" in config:
        return env
    print("")
    print(20 * "#" + " WRAPPING ENV " + 20 * "#")
    for wrapper, kwargs in config["wrapper"].items():
        # try:
        env = skript_globals[wrapper](env, **kwargs)
        print("Wrapped env with {} wrapper!".format(wrapper))
        print("")
        # except Exception as e:
        #     raise RuntimeError("Wrapper class {} was not found. Perhaps it is not imported! Error: {}".format(wrapper, e))
    return env


def get_yaml_from_run_string(s: str) -> str:
    """
    ...assuming that wandb will not assign any more underscores in the run_name and yaml and run_name are
    separated by an underscore.
    """
    c = "_"
    c_ids = [pos for pos, char in enumerate(s) if char == c]
    separator_id = c_ids[-1]
    return s[:separator_id] + ".yml"