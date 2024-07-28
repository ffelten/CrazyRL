"""Catch environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point trying to escape."""

import time
from typing import Optional
from typing_extensions import override

import numpy as np
from gymnasium import spaces

from crazy_rl.multi_agent.numpy.base_parallel_env import (
    CLOSENESS_THRESHOLD,
    BaseParallelEnv,
    _distance_to_target,
)


class Catch(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target trying to escape."""

    metadata = {"render_modes": ["human", "real", "unity"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        drone_ids: np.ndarray,
        init_flying_pos: np.ndarray,
        init_target_location: np.ndarray,
        target_speed: float,
        target_id: Optional[int] = None,
        render_mode=None,
        size: int = 2,
        multi_obj: bool = False,
        swarm=None,
    ):
        """Catch environment for Crazyflies 2.

        Args:
            drone_ids: Array of drone ids
            init_flying_pos: Array of initial positions of the drones when they are flying
            init_target_location: Array of the initial position of the moving target
            target_speed: Distance traveled by the target at each timestep
            target_id: Target id if you want a real target
            render_mode: Render mode: "human", "real" or None
            size: Size of the map
            multi_obj: Whether to return a multi-objective reward
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
        self.multi_obj = multi_obj
        self.size = size

        super().__init__(
            render_mode=render_mode,
            size=size,
            init_flying_pos=self._init_flying_pos,
            target_location=self._target_location,
            agents_names=self._agents_names,
            drone_ids=drone_ids,
            target_id=target_id,
            swarm=swarm,
        )

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=np.tile(np.array([-self.size, -self.size, 0], dtype=np.float32), self.num_drones + 1),
            high=np.tile(np.array([self.size, self.size, 3], dtype=np.float32), self.num_drones + 1),
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
            dtype=np.float32,
        )

    @override
    def _action_space(self, agent):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    def _compute_obs(self):
        obs = dict()

        for agent in self._agents_names:
            obs[agent] = self._agent_location[agent].copy()
            obs[agent] = np.append(obs[agent], self._target_location["unique"])

            for other_agent in self._agents_names:
                if other_agent != agent:
                    obs[agent] = np.append(obs[agent], self._agent_location[other_agent])

        return obs

    def _move_target(self):
        # mean of the agent's positions
        mean = np.array([0, 0, 0])
        for agent in self.agents:
            mean = mean + self._agent_location[agent]

        mean = mean / self.num_drones

        dist = np.linalg.norm(mean - self._target_location["unique"])
        self._target_location["unique"] = self._target_location["unique"].copy()

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
            [self.size, self.size, 3],
            out=self._target_location["unique"],
        )

    @override
    def _transition_state(self, actions):
        target_point_action = dict()
        state = self._agent_location

        # new targets
        self._previous_target = self._target_location.copy()
        self._move_target()

        for agent in self.agents:
            # Actions are clipped to stay in the map and scaled to do max 20cm in one step
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent] * 0.2, [-self.size, -self.size, 0], [self.size, self.size, 3]
            )

        return target_point_action

    @override
    def _compute_reward(self):
        # Reward is the mean distance to the other agents minus the distance to the target
        reward = dict()

        for agent in self._agents_names:
            reward_far_from_other_agents = 0
            reward_close_to_target = 0

            # mean distance to the other agents
            for other_agent in self._agents_names:
                if other_agent != agent:
                    reward_far_from_other_agents += np.linalg.norm(
                        self._agent_location[agent] - self._agent_location[other_agent]
                    )

            reward_far_from_other_agents /= self.num_drones - 1

            # distance to the target
            # (!) targets and locations must be updated before this
            dist_from_old_target = _distance_to_target(self._agent_location[agent], self._previous_target["unique"])
            old_dist = _distance_to_target(self._previous_location[agent], self._previous_target["unique"])

            # reward should be new_potential - old_potential but since the distances should be negated we reversed the signs
            # -new_potential - (-old_potential) = old_potential - new_potential
            reward_close_to_target = old_dist - dist_from_old_target

            # collision between two drones
            for other_agent in self._agents_names:
                if other_agent != agent and (
                    np.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent]) < CLOSENESS_THRESHOLD
                ):
                    reward_far_from_other_agents = -10
                    reward_close_to_target = -10

            # collision with the ground or the target
            if (
                self._agent_location[agent][2] < CLOSENESS_THRESHOLD
                or np.linalg.norm(self._agent_location[agent] - self._target_location["unique"]) < CLOSENESS_THRESHOLD
            ):
                reward_far_from_other_agents = -10
                reward_close_to_target = -10

            if self.multi_obj:
                reward[agent] = np.array([reward_close_to_target, reward_far_from_other_agents])
            else:
                # MO reward linearly combined using hardcoded weights
                reward[agent] = 0.9995 * reward_close_to_target + 0.0005 * reward_far_from_other_agents

        return reward

    @override
    def _compute_terminated(self):
        terminated = dict()

        for agent in self.agents:
            terminated[agent] = False

        for agent in self.agents:
            # collision between two drones
            for other_agent in self.agents:
                if other_agent != agent:
                    terminated[agent] = terminated[agent] or (
                        np.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent]) < CLOSENESS_THRESHOLD
                    )

            # collision with the ground
            terminated[agent] = terminated[agent] or (self._agent_location[agent][2] < CLOSENESS_THRESHOLD)

            # collision with the target
            terminated[agent] = terminated[agent] or (
                np.linalg.norm(self._agent_location[agent] - self._target_location["unique"]) < CLOSENESS_THRESHOLD
            )

            if terminated[agent] and self.render_mode != "real":
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

    @override
    def state(self):
        return np.append(np.array(list(self._agent_location.values())).flatten(), self._target_location["unique"])


if __name__ == "__main__":
    parallel_env = Catch(
        drone_ids=np.array([0, 1, 2, 3]),
        render_mode="unity",
        init_flying_pos=np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [2, 2, 1]]),
        init_target_location=np.array([1, 1, 2.5]),
        target_speed=0.1,
    )

    observations, infos = parallel_env.reset()

    while parallel_env.agents:
        actions = {
            agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        parallel_env.render()
        time.sleep(0.5)
        print("obs", observations, "reward", rewards)
        time.sleep(0.02)
