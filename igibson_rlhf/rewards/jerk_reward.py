import numpy as np
from igibson.reward_functions.reward_function_base import BaseRewardFunction


class JerkReward(BaseRewardFunction):
    def __init__(self, config):
        super(JerkReward, self).__init__(config)
        self.control_frequency = self.config.get("control_frequency", 5)
        self.jerk_weight = self.config.get("jerk_reward_weight", 0.1)
        self.reward_value = self.config.get("jerk_reward_value", 1)
        self.action_history = []  # to store last three actions
        self.max_jerk = 0.0  # to track max jerk during an episode

    def get_reward(self, task, env):
        # Add the current action to the history (keeping only the last 3)
        self.action_history.append(np.array(env.action_last))
        if len(self.action_history) < 3:
            return 0  # Not enough data to compute jerk yet

        # Keep only the last three actions in the history
        if len(self.action_history) > 3:
            self.action_history.pop(0)

        # Compute the jerk using finite difference [1, -2, 1]
        a_t = self.action_history[-1]      # most recent action
        a_t1 = self.action_history[-2]     # previous action
        a_t2 = self.action_history[-3]     # second previous action
        jerk = (a_t - 2 * a_t1 + a_t2) * self.control_frequency**2

        # Compute the magnitude of the jerk (sum of squares across action dimensions)
        jerk_magnitude = np.sum(np.square(jerk))

        # Update the max jerk observed
        self.max_jerk = max(self.max_jerk, jerk_magnitude)

        # Compute normalized jerk penalty
        return (
            -self.reward_value
            * self.jerk_weight
            * (jerk_magnitude / self.max_jerk)
        )

    def reset(self, task, env):
        self.action_history = []