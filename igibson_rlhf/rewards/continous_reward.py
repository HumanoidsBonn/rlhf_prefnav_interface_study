from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class ContinuousReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(ContinuousReward, self).__init__(config)
        self.continuous_reward = self.config.get("continuous_reward", -0.001)

    def get_reward(self, task, env):
        return self.continuous_reward
