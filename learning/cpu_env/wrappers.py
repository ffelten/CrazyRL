from typing import Optional

import numpy as np
import pettingzoo
from gymnasium.wrappers.normalize import RunningMeanStd
from pettingzoo.utils import BaseParallelWrapper


class NormalizeReward(BaseParallelWrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: pettingzoo.ParallelEnv,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        BaseParallelWrapper.__init__(self, env)

        self.return_rms = [RunningMeanStd(shape=()) for _ in self.possible_agents]
        self.returns = np.zeros(len(self.possible_agents))
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        for i, agent in enumerate(self.possible_agents):
            self.returns[i] = self.returns[i] * self.gamma * (1 - terminateds[agent]) + rews[agent]
            rews[agent] = self._normalize(rews[agent], i)
        return obs, rews, terminateds, truncateds, infos

    def _normalize(self, rews, i):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms[i].update(np.array([self.returns[i]]))
        return rews / np.sqrt(self.return_rms[i].var + self.epsilon)


class RecordEpisodeStatistics(BaseParallelWrapper):
    """This wrapper will record episode statistics and print them at the end of each episode."""

    def __init__(self, env: pettingzoo.ParallelEnv):
        """This wrapper will record episode statistics and print them at the end of each episode.

        Args:
            env (env): The environment to apply the wrapper
        """
        BaseParallelWrapper.__init__(self, env)
        self.episode_rewards = {agent: 0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}

    def step(self, action):
        """Steps through the environment, recording episode statistics."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] += rews[agent]
            self.episode_lengths[agent] += 1
        if all(terminateds.values()) or all(truncateds.values()):
            infos["episode"] = {
                "r": self.episode_rewards,
                "l": self.episode_lengths,
            }
        return obs, rews, terminateds, truncateds, infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment, recording episode statistics."""
        obs, info = self.env.reset(seed=seed, options=options)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] = 0
            self.episode_lengths[agent] = 0
        return obs, info
