from gym.wrappers import RecordVideo
from stable_baselines3.common.callbacks import EvalCallback
from sb3_extensions.common.evaluation_igibson_enhanced import evaluate_policy_igibson
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import os
import warnings
import numpy as np
import time

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

class EvalCallbackIGibson(EvalCallback):
    def __init__(
            self,
            *args,
            record_eval=False,
            video_folder=None,
            **kwargs,
                 ):
        super(EvalCallbackIGibson, self).__init__(*args, **kwargs)

        if record_eval:
            assert video_folder is not None, "You must provide a video folder when render_eval=True"
            if isinstance(self.eval_env, (DummyVecEnv, VecEnv)):
                env = self.eval_env.envs[0]
            else:
                env = self.eval_env
            self.eval_env = RecordVideo(
                env,
                video_folder=video_folder,
                # step_trigger=lambda step: True,
                # TODO Overwrites video files in folder, if continue training:
                episode_trigger=lambda episode_id: episode_id % self.n_eval_episodes == 0,
                name_prefix="model_eval",
                # merge_n_videos=self.n_eval_episodes,
            )

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            time_start = time.time()
            episode_rewards, episode_lengths, additional_info = evaluate_policy_igibson(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            time_eval = time.time() - time_start

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("time/episodes", self.model._episode_num, exclude="stdout")
            # self.logger.record("time/total_timesteps", self.model.num_timesteps, exclude="stdout")

            # =========================================================================================================
            # CUSTOM LOGGING OF STUFF
            # =========================================================================================================
            self.logger.record("eval/total_time", float(time_eval))
            if hasattr(self.eval_env, "action_timestep"):
                self.logger.record("eval/real_time_factor", float(mean_ep_length * self.n_eval_episodes
                                                                  * self.eval_env.action_timestep / time_eval))
            for key in additional_info:
                self.logger.record(key, float(additional_info[key]))


            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
