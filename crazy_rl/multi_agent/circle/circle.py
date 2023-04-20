"""Circle environment for Crazyflie 2. Each agent is supposed to learn to perform a circle around a target point."""
import time
from typing import List
from typing_extensions import override

import numpy as np
from gymnasium import spaces
from pettingzoo.test.parallel_test import parallel_api_test

from crazy_rl.multi_agent.base_parallel_env import BaseParallelEnv


class Circle(BaseParallelEnv):
    """A Parallel Environment where drone learn how to perform a circle."""

    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        drone_ids: np.ndarray[int],
        init_flying_pos: np.ndarray[int],
        render_mode=None,
        num_intermediate_points: int = 10,
        size: int = 4,
        swarm=None,
    ):
        """Circle environment for Crazyflies 2.

        Args:
            drone_ids: Array of drone ids
            init_flying_pos: Array of initial positions of the drones when they are flying
            render_mode: Render mode: "human", "real" or None
            num_intermediate_points: Number of intermediate points in the target circle
            size: Size of the map
            swarm: Swarm object, used for real tests. Ignored otherwise.
        """
        self.num_drones = len(drone_ids)

        self._agent_location = dict()
        self._target_location = dict()
        self._init_flying_pos = dict()
        self._agents_names = np.array(["agent_" + str(i) for i in drone_ids])
        self.timestep = 0

        circle_radius = 0.5  # [m]
        self.num_intermediate_points = num_intermediate_points
        # Ref is a list of 2d arrays for each agent
        # each 2d array contains the reference points (xyz) for the agent at each timestep
        self.ref: List[np.ndarray] = []

        for i, agent in enumerate(self._agents_names):
            self._init_flying_pos[agent] = init_flying_pos[i].copy()

            ts = 2 * np.pi * np.arange(num_intermediate_points) / num_intermediate_points

            self.ref.append(np.zeros((num_intermediate_points, 3)))
            self.ref[i][:, 2] = circle_radius * np.sin(ts) + (init_flying_pos[i][2])  # z-position
            self.ref[i][:, 1] = init_flying_pos[i][1]  # y-position
            self.ref[i][:, 0] = circle_radius * (1 - np.cos(ts)) + (init_flying_pos[i][0] - circle_radius)  # x-position

        self._agent_location = self._init_flying_pos.copy()
        self._target_location = self._init_flying_pos.copy()

        self.size = size

        super().__init__(
            render_mode=render_mode,
            size=size,
            init_flying_pos=self._init_flying_pos,
            target_location=self._target_location,
            agents_names=self._agents_names,
            drone_ids=drone_ids,
            swarm=swarm,
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
        for i, agent in enumerate(self._agents_names):
            t = self.timestep % self.num_intermediate_points  # redo the circle if the end is reached
            self._target_location[agent] = self.ref[i][t]
            obs[agent] = np.hstack([self._agent_location[agent], self._target_location[agent]]).reshape(
                6,
            )
        return obs

    @override
    def _compute_action(self, actions):
        target_point_action = dict()
        state = self._get_drones_state()

        for agent in self._agents_names:
            # Actions are clipped to stay in the map and scaled to do max 20cm in one step
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent] * 0.2, [-self.size - 1, -self.size - 1, 0], self.size - 1
            )

        return target_point_action

    @override
    def _compute_reward(self):
        # Reward is based on the euclidean distance to the target point
        reward = dict()
        for agent in self._agents_names:
            reward[agent] = -1 * np.linalg.norm(self._target_location[agent] - self._agent_location[agent])
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
        Circle(
            drone_ids=np.array([0, 1]),
            render_mode=None,
            init_flying_pos=np.array([[0, 0, 1], [1, 1, 1]]),
        ),
        num_cycles=10,
    )

    parallel_env = Circle(
        drone_ids=np.array([0, 1]),
        render_mode="human",
        init_flying_pos=np.array([[0, 0, 1], [2, 2, 1]]),
    )

    observations = parallel_env.reset()

    while parallel_env.agents:
        actions = {
            agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        parallel_env.render()
        print("obs", observations, "reward", rewards)
        time.sleep(0.2)
