# A custom callback
from stable_baselines3.common.callbacks import BaseCallback


class iGibsonLoggerCallback(BaseCallback):
    """ Logs the net change in cash between the beginning and end of each epoch/run. """

    def __init__(self, verbose=0):
        super(iGibsonLoggerCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_rollout_end(self) -> None:
        self.logger.record("time/episodes", self.model._episode_num, exclude="stdout")
        self.logger.record("time/total_timesteps", self.model.num_timesteps, exclude="stdout")
        # if "time/fps" in self.logger.name_to_value.keys():
        #     # if getattr(self.training_env, "action_timestep") is not None:
        #     fps = self.logger.name_to_value["time/fps"]
        #     rtf = fps * self.training_env.action_timestep
        #     self.logger.record("time/real_time_factor", rtf, exclude="stdout")
        return True

    def _on_step(self) -> bool:
        return True