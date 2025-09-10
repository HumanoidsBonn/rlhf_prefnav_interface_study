from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class CycleReward(BaseRewardFunction):
    """
    CycleReward / Penalty for visiting a square more than once
    """

    def __init__(self, config):
        super(CycleReward, self).__init__(config)
        self.positions = {}

    def reset(self, task, env):
        """
        Reward function-specific reset

        :param task: task instance
        :param env: environment instance
        """
        self.positions = {}
        return
    
    def get_reward(self, task, env):
        current_position = env.robots[0].get_position()
        current_x = current_position[0] // 0.01
        current_y = current_position[1] // 0.01

        if current_x in self.positions:
            if current_y in self.positions[current_x]:
                if env.current_step - self.positions[current_x][current_y] > 5:
                    return -2
            else:
                self.positions[current_x][current_y] = env.current_step
        else:
            self.positions[current_x] = { current_y: env.current_step }
        return 0
