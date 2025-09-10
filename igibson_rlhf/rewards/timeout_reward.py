from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class TimeoutReward(BaseRewardFunction):
    def __init__(self, config):
        super(TimeoutReward, self).__init__(config)
        self.timeout_reward = self.config.get("timeout_reward", -1.0)
        self.max_step = self.config.get("max_step", 500)

    def get_reward(self, task, env):
        done = env.current_step >= self.max_step
        reward = self.timeout_reward if done else 0.0
        return reward
