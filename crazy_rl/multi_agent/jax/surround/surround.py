"""Surround environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point."""
import time

# from functools import partial
from typing_extensions import override

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv


class Surround(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a target point."""

    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        target_location: jnp.ndarray,
        render_mode=None,
        size: int = 4,
        swarm=None,
    ):
        """Surround environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            target_location: Array of the position of the target point
            render_mode: Render mode: "human", "real" or None
            size: Size of the map
            swarm: Swarm object, used for real tests. Ignored otherwise.
        """
        self.num_drones = num_drones

        self.alive_agents = jnp.array([i for i in range(self.num_drones)])

        self._target_location = target_location  # unique target location for all agents

        self._init_flying_pos = init_flying_pos

        self._agent_location = jnp.copy(init_flying_pos)

        self.timestep = 0

        self.size = size

        super().__init__(
            render_mode=render_mode,
            size=size,
            init_flying_pos=self._init_flying_pos,
            target_location=self._target_location,
            num_drones=self.num_drones,
            swarm=swarm,
        )

    @override
    @jit  # fonctionne avec jit
    def _observation_space(self, agent):
        return spaces.Box(
            low=np.tile(np.array([-self.size, -self.size, 0], dtype=np.float32), self.num_drones + 1),
            high=np.tile(np.array([self.size, self.size, self.size], dtype=np.float32), self.num_drones + 1),
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
            dtype=jnp.float32,
        )

    @override  # ne fonctionne pas avec jit (je crois que j'ai pas compris en fait)
    def _action_space(self):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override  # ne fonctionne pas avec jit
    def _compute_obs(self):
        obs = jnp.array([[] for _ in range(self.num_drones)])

        for agent in range(self.num_drones):
            obs_agent = jnp.append(self._agent_location[agent], self._target_location)

            for other_agent in range(self.num_drones):
                if other_agent != agent:
                    obs_agent = jnp.append(obs_agent, self._agent_location[other_agent])

            obs = obs.at[agent].set(obs_agent)

        return obs

    @override  # nope
    def _compute_action(self, actions):
        target_point_action = np.zeros((self.num_drones, 3))
        state = self._get_drones_state()

        for agent in self.alive_agents:
            # Actions are clipped to stay in the map and scaled to do max 20cm in one step
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent] * 0.2, jnp.array([-self.size, -self.size, 0]), self.size
            )

        return target_point_action

    @override  # nope
    def _compute_reward(self, crash, end):
        # Reward is the mean distance to the other agents minus the distance to the target
        rewards = jnp.zeros(self.num_drones)

        for agent in range(self.num_drones):
            reward = 0
            if end:
                # mean distance to the other agents
                for other_agent in range(self.num_drones):
                    if other_agent != agent:
                        reward += jnp.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent])

                reward /= self.num_drones - 1

                reward *= 0.05

                # a maximum value minus the distance to the target
                reward += 0.95 * (2 * self.size - jnp.linalg.norm(self._agent_location[agent] - self._target_location))

                rewards = rewards.at[agent].set(reward)

        rewards = rewards + crash * -10 * jnp.ones(self.num_drones)

        """
        # collision between two drones
        for other_agent in range(self.num_drones):
            if other_agent != agent and (
                jnp.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent]) < 0.2
            ):
                reward -= 100

        # collision with the ground
        if self._agent_location[agent][2] < 0.2:
            reward -= 100

        # collision with the target
        if jnp.linalg.norm(self._agent_location[agent] - self._target_location) < 0.2:
            reward -= 100
        """

        return rewards

    @override  # nope
    # @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, alive_agents, crash, end):
        # terminated = jnp.array([False for _ in range(self.num_drones)])
        # agents = jnp.copy(self.alive_agents)
        crash = False

        # End of the game
        if self.timestep == 100:
            terminated = jnp.array([True for _ in range(self.num_drones)])
            alive_agents = jnp.array([])
            end = True

        else:
            # collision with the ground and the target
            terminated = jnp.logical_or(
                self._agent_location[:, 2] < 0.2, jnp.linalg.norm(self._agent_location - self._target_location) < 0.2
            )

            for agent in alive_agents:
                terminated_agent = False
                # collision between two drones
                for other_agent in alive_agents:
                    terminated_agent = terminated_agent or (
                        other_agent != agent
                        and jnp.linalg.norm(self._agent_location[agent] - self._agent_location[other_agent]) < 0.2
                    )
                terminated = terminated.at[agent].set(terminated_agent)

            if jnp.any(terminated):
                terminated = jnp.array([True for _ in range(self.num_drones)])
                alive_agents = jnp.array([])
                crash = True

        return terminated, alive_agents, crash, end

    @override  # nope
    def _compute_truncation(self):
        if self.timestep == 200:
            truncation = {agent: True for agent in range(self.num_drones)}
            self.alive_agents = jnp.array([])
            self.timestep = 0  # pareil ça c'est modifié
        else:
            truncation = {agent: False for agent in range(self.num_drones)}
        return truncation

    @override  # nope
    def _compute_info(self):
        info = jnp.array([])
        return info


if __name__ == "__main__":
    parallel_env = Surround(
        num_drones=5,
        render_mode="human",
        init_flying_pos=jnp.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
    )

    observations = parallel_env.reset()

    global_step = 0
    start_time = time.time()
    for i in range(1000):
        while parallel_env.alive_agents.size > 0:
            actions = jnp.array([parallel_env.action_space().sample() for agent in parallel_env.alive_agents])
            # this is where you would insert your policy
            observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
            parallel_env.render()

            # print("obs", observations, "reward", rewards)
            # print("reward", rewards)

            if global_step % 100 == 0:
                print("SPS:", int(global_step / (time.time() - start_time)))

            global_step += 1

            # time.sleep(0.02)
        # print("alive_agents", parallel_env.alive_agents)
        observations = parallel_env.reset()
