"""Hover environment for Crazyflies 2."""
import time
from typing import Dict, List
from typing_extensions import override

import numpy as np
from gymnasium import spaces
from pettingzoo.test.parallel_test import parallel_api_test

from crazy_rl.multi_agent.base_parallel_env import BaseParallelEnv


class Hover(BaseParallelEnv):
    """A Parallel Environment where drone learn how to hover around a target point."""

    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        drone_ids: List[int],
        init_xyzs: List[List[int]],
        init_target_points: List[List[int]],
        render_mode=None,
        size: int = 4,
    ):
        """Hover environment for Crazyflies 2.

        Args:
            drone_ids: List of drone ids
            init_xyzs: List of initial positions of the drones
            init_target_points: List of initial target points of the drones
            render_mode: Render mode: "human", "real" or None
            size: Size of the map
        """
        self.num_drones = len(drone_ids)

        self._agent_location = dict()
        self._target_location = dict()
        self._init_target_points = dict()
        self._init_xyzs = dict()
        self._agents_names = ["agent_" + str(i) for i in drone_ids]
        self.timestep = 0

        for i, agent in enumerate(self._agents_names):
            if init_xyzs is None:
                self._init_xyzs[agent] = np.zeros(3, dtype=float)
            else:
                self._init_xyzs[agent] = init_xyzs[i]
            if init_target_points is None:
                self._init_target_points[agent] = np.zeros(3, dtype=float)
            else:
                self._init_target_points[agent] = init_target_points[i]

        self._agent_location = self._init_xyzs.copy()
        self._target_location = self._init_target_points.copy()

        self.size = size

        super().__init__(
            render_mode=render_mode,
            size=size,
            agents_names=self._agents_names,
            init_xyzs=self._init_xyzs,
            init_target_points=self._init_target_points,
            drone_ids=drone_ids,
        )

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=np.array([-self.size - 1, -self.size - 1, 0, -self.size - 1, -self.size - 1, 0], dtype=np.float32),
            high=np.array(
                [self.size - 1, self.size - 1, self.size - 1, self.size - 1, self.size - 1, self.size - 1], dtype=np.float32
            ),
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
    def _compute_action(self, actions: Dict[str, np.ndarray]):
        target_point_action = dict()
        state = self._get_drones_state()
        for agent in self._agents_names:
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent], [-self.size - 1, -self.size - 1, 0], self.size - 1
            )
        return target_point_action

    @override
    def _compute_reward(self):
        reward = dict()
        for agent in self._agents_names:
            reward[agent] = -1 * np.linalg.norm(self._target_location[agent] - self._agent_location[agent]) ** 2
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


if __name__ == "__main__":

    parallel_api_test(
        Hover(
            drone_ids=[1, 2],
            render_mode=None,
            init_xyzs=[[0, 0, 0], [1, 1, 0]],
            init_target_points=[[0, 0, 1], [1, 1, 1]],
        ),
        num_cycles=10,
    )

    parallel_env = Hover(
        drone_ids=[1, 2],
        render_mode="human",
        init_xyzs=[[0, 0, 0], [1, 1, 0]],
        init_target_points=[[0, 0, 1], [1, 1, 1]],
    )

    observations = parallel_env.reset()

    while parallel_env.agents:
        actions = {
            agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        print("obs", observations, "reward", rewards)
        time.sleep(1)
