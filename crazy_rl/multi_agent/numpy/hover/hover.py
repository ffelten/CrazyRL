"""Hover environment for Crazyflies 2."""
import time
from typing import Dict
from typing_extensions import override

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo.test.parallel_test import parallel_api_test

from crazy_rl.multi_agent.numpy.base_parallel_env import (
    BaseParallelEnv,
    _distance_to_target,
)


class Hover(BaseParallelEnv):
    """A Parallel Environment where drone learn how to hover around a target point."""

    metadata = {"render_modes": ["human", "real", "unity"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        drone_ids: npt.NDArray[int],
        init_flying_pos: npt.NDArray[int],
        render_mode=None,
        size: int = 2,
    ):
        """Hover environment for Crazyflies 2.

        Args:
            drone_ids: Array of drone ids
            init_flying_pos: Array of initial positions of the drones when they are flying
            render_mode: Render mode: "human", "real" or None
            size: Size of the map
        """
        self.num_drones = len(drone_ids)

        self._agent_location = dict()
        self._target_location = dict()
        self._init_flying_pos = dict()
        self._agents_names = np.array(["agent_" + str(i) for i in drone_ids])
        self.timestep = 0

        for i, agent in enumerate(self._agents_names):
            self._init_flying_pos[agent] = init_flying_pos[i].copy()

        self._agent_location = self._init_flying_pos.copy()
        self._target_location = self._init_flying_pos.copy()

        self.size = size

        super().__init__(
            render_mode=render_mode,
            size=size,
            agents_names=self._agents_names,
            init_flying_pos=self._init_flying_pos,
            target_location=self._target_location,
            drone_ids=drone_ids,
            target_id=None,  # Should be none for multi target envs
        )

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=np.array([-self.size, -self.size, 0, -self.size, -self.size, 0], dtype=np.float32),
            high=np.array([self.size, self.size, 3, self.size, self.size, 3], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

    @override
    def _action_space(self, agent):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    def _compute_obs(self):
        obs = dict()
        for agent in self._agents_names:
            obs[agent] = np.hstack([self._agent_location[agent], self._target_location[agent]]).reshape(
                6,
            )
        return obs

    @override
    def _transition_state(self, actions: Dict[str, np.ndarray]):
        target_point_action = dict()
        state = self._agent_location
        for agent in self._agents_names:
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent], [-self.size, -self.size, 0], [self.size, self.size, 3]
            )
        return target_point_action

    @override
    def _compute_reward(self):
        # Reward is based on the euclidean distance to the target point
        reward = dict()
        for i, agent in enumerate(self._agents_names):
            # (!) targets and locations must be updated before this
            dist_from_target = _distance_to_target(self._agent_location[agent], self._target_location[agent])
            old_dist = _distance_to_target(self._previous_location[agent], self._target_location[agent])

            # reward should be new_potential - old_potential but since the distances should be negated we reversed the signs
            # -new_potential - (-old_potential) = old_potential - new_potential
            reward[agent] = old_dist - dist_from_target
        return reward

    @override
    def _compute_terminated(self):
        return {agent: False for agent in self._agents_names}

    @override
    def _compute_truncation(self):
        if self.timestep == 200:
            truncation = {agent: True for agent in self._agents_names}
            self.agents = []
            self.timestep = 0
        else:
            truncation = {agent: False for agent in self._agents_names}

        return truncation

    @override
    def _compute_info(self):
        info = dict()
        for agent in self._agents_names:
            info[agent] = {"distance": np.linalg.norm(self._agent_location[agent] - self._target_location[agent], ord=1)}
        return info

    @override
    def state(self):
        return np.append(np.array(list(self._agent_location.values())).flatten(), self._target_location["unique"])


if __name__ == "__main__":
    parallel_api_test(
        Hover(
            drone_ids=np.array([0, 1]),
            render_mode=None,
            init_flying_pos=np.array([[0, 0, 1], [1, 1, 1]]),
        ),
        num_cycles=10,
    )

    parallel_env = Hover(
        drone_ids=np.array([0, 1]),
        render_mode="unity",
        init_flying_pos=np.array([[0, 0, 1], [1, 1, 1]]),
    )

    observations, infos = parallel_env.reset()

    while parallel_env.agents:
        actions = {
            agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        time.sleep(0.2)
        print("obs", observations, "reward", rewards)
        time.sleep(0.02)
