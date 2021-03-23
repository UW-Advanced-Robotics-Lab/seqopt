import os
from typing import Callable, Optional, Union
import warnings

import gym
import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from seqopt.fsm.evaluation import evaluate_policy


class OptionsEvalCallback(EvalCallback):

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            reward_func: Callable,
            callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            best_model_save_path: str = None,
            deterministic_actions: bool = True,
            deterministic_transitions: bool = True,
            render: bool = False,
            verbose: int = 1,
    ):
        super(OptionsEvalCallback, self).__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic_actions,  # We reuse the deterministic variable of the base class for the action
            render=render,
            verbose=verbose
        )

        self.reward_func = reward_func
        self.deterministic_transitions = deterministic_transitions

    def init_callback(self, model: "seqopt.algorithms.SequenceSAC") -> None:
        self.model = model
        self.training_env = model.get_env()
        self.logger = logger
        self._init_callback()

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                reward_func=self.reward_func,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic_actions=self.deterministic,
                deterministic_terminations=self.deterministic_transitions,
                return_episode_rewards=True,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            min_reward, max_reward = np.min(episode_rewards), np.max(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            print(f"Rewards: {episode_rewards}, Mean Reward: {mean_reward}, Std Reward: {std_reward}")

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}," f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/min_reward", min_reward)
            self.logger.record("eval/max_reward", max_reward)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True
