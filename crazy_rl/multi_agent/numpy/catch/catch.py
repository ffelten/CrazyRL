"""Catch environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point trying to escape."""

import time
from typing_extensions import override

import numpy as np
from gymnasium import spaces

from crazy_rl.multi_agent.numpy.base_parallel_env import BaseParallelEnv


class Catch(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target trying to escape."""

    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        drone_ids: np.ndarray,
        init_flying_pos: np.ndarray,
        init_target_location: np.ndarray,
        target_speed: float,
        render_mode=None,
        size: int = 4,
        swarm=None,
    ):
        """Catch environment for Crazyflies 2.

        Args:
            drone_ids: Array of drone ids
            init_flying_pos: Array of initial positions of the drones when they are flying
            init_target_location: Array of the initial position of the moving target
            target_speed: Distance traveled by the target at each timestep
            render_mode: Render mode: "human", "real" or None
            size: Size of the map
            swarm: Swarm object, used for real tests. Ignored otherwise.
        """
        self.num_drones = len(drone_ids)

        self._agent_location = dict()

        self._target_location = {"unique": init_target_location}  # unique target location for all agents

        self.target_speed = target_speed

        self._init_flying_pos = dict()
        self._agents_names = np.array(["agent_" + str(i) for i in drone_ids])
        self.timestep = 0

        for i, agent in enumerate(self._agents_names):
            self._init_flying_pos[agent] = init_flying_pos[i].copy()

        self._agent_location = self._init_flying_pos.copy()

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
            low=np.tile(np.array([-self.size, -self.size, 0], dtype=np.float32), self.num_drones + 1),
            high=np.tile(np.array([self.size, self.size, self.size], dtype=np.float32), self.num_drones + 1),
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
            dtype=np.float32,
        )

    @override
    def _action_space(self, agent):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    def _compute_obs(self):
        obs = dict()

        # mean of the agent's positions
        mean = np.array([0, 0, 0])

        for agent in self.agents:
            mean = mean + self._agent_location[agent]

        mean = mean / self.num_drones

        dist = np.linalg.norm(mean - self._target_location["unique"])

        # go to the opposite direction of the mean of the agents
        if dist > 0.2:
            self._target_location["unique"] += (self._target_location["unique"] - mean) / dist * self.target_speed

        # if the mean of the agents is too close to the target, move the target in a random direction, slowly because
        # it hesitates
        else:
            self._target_location["unique"] += np.random.random_sample(3) * self.target_speed * 0.1

        # if the target is out of the map, put it back in the map
        np.clip(
            self._target_location["unique"],
            [-self.size, -self.size, 0.2],
            [self.size, self.size, self.size],
            out=self._target_location["unique"],
        )

        for agent in self._agents_names:
            obs[agent] = self._agent_location[agent].copy()
            obs[agent] = np.append(obs[agent], self._target_location["unique"])

            for other_agent in self._agents_names:
                if other_agent != agent:
                    obs[agent] = np.append(obs[agent], self._agent_location[other_agent])

        return obs

    @override
    def _compute_action(self, actions):
        target_point_action = dict()
        state = self._get_drones_state()

        for agent in self.agents:
            # Actions are clipped to stay in the map and scaled to do max 20cm in one step
            target_point_action[agent] = np.clip(state[agent] + actions[agent] * 0.2, [-self.size, -self.size, 0], self.size)

        return target_point_action

    @override
    def _compute_reward(self):
        # Reward is the mean distance to the other agents minus the distance to the target
        reward = dict()

        for agent in self._agents_names:
            reward[agent] = 0

            # mean distance to the other agents
            for other_agent in self._agents_names:
                if other_agent != agent:
                    reward[agent] += np.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent])

            reward[agent] /= self.num_drones - 1
            reward[agent] *= 0

            # distance to the target
            reward[agent] -= 1 * np.linalg.norm(self._agent_location[agent] - self._target_location["unique"])

            # collision between two drones
            for other_agent in self._agents_names:
                if other_agent != agent and (
                    np.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent]) < 0.2
                ):
                    reward[agent] -= 100

            # collision with the ground
            if self._agent_location[agent][2] < 0.2:
                reward[agent] -= 100

            # collision with the target
            if np.linalg.norm(self._agent_location[agent] - self._target_location["unique"]) < 0.2:
                reward[agent] -= 100

        return reward

    @override
    def _compute_terminated(self):
        terminated = dict()

        for agent in self.agents:
            terminated[agent] = False

            # collision between two drones
            for other_agent in self.agents:
                if other_agent != agent:
                    terminated[agent] = terminated[agent] or (
                        np.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent]) < 0.2
                    )

            # collision with the ground
            terminated[agent] = terminated[agent] or (self._agent_location[agent][2] < 0.2)

            # collision with the target
            terminated[agent] = terminated[agent] or (
                np.linalg.norm(self._agent_location[agent] - self._target_location["unique"]) < 0.2
            )

            if terminated[agent]:
                for other_agent in self.agents:
                    terminated[other_agent] = True
                self.agents = []

        return terminated

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
        return info


if __name__ == "__main__":
    parallel_env = Catch(
        drone_ids=np.array([0, 1, 2, 3]),
        render_mode="human",
        init_flying_pos=np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [2, 2, 1]]),
        init_target_location=np.array([1, 1, 2.5]),
        target_speed=0.1,
    )

    observations = parallel_env.reset()

    while parallel_env.agents:
        actions = {
            agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        parallel_env.render()
        print("obs", observations, "reward", rewards)
        time.sleep(0.02)
