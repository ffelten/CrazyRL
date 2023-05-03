"""Surround environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point."""
import time
from functools import partial
from typing_extensions import override

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit, vmap

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

        self._target_location = target_location  # unique target location for all agents

        self._init_flying_pos = init_flying_pos

        self._agent_location = jnp.copy(init_flying_pos)

        self.timestep = 0

        self.size = size

        self.norm = vmap(jnp.linalg.norm)

        super().__init__(
            render_mode=render_mode,
            size=size,
            init_flying_pos=self._init_flying_pos,
            target_location=self._target_location,
            num_drones=self.num_drones,
            swarm=swarm,
        )

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=np.tile(np.array([-self.size, -self.size, 0], dtype=np.float32), self.num_drones + 1),
            high=np.tile(np.array([self.size, self.size, self.size], dtype=np.float32), self.num_drones + 1),
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
            dtype=jnp.float32,
        )

    @override
    def _action_space(self):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, agent_location):
        obs = jnp.array([[] for _ in range(self.num_drones)])

        for agent in range(self.num_drones):
            obs_agent = jnp.append(agent_location[agent], self._target_location)

            for other_agent in range(self.num_drones):
                if other_agent != agent:
                    obs_agent = jnp.append(obs_agent, agent_location[other_agent])

            obs = obs.at[agent].set(obs_agent)

        return obs

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_action(self, actions, state):
        # Actions are clipped to stay in the map and scaled to do max 20cm in one step
        target_point_action = jnp.clip(state + actions * 0.2, jnp.array([-self.size, -self.size, 0]), self.size)

        return target_point_action

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, end, crash, agent_location):
        # Reward is the mean distance to the other agents minus the distance to the target
        """
        rewards = jnp.zeros(self.num_drones)

        for agent in range(self.num_drones):

            rewards = rewards.at[agent].set(jnp.sum(self.norm(self._agent_location[agent] - self._agent_location)))
        """
        # a maximum value minus the distance to the target
        rewards = end * (
            # mean distance to the other agents
            jnp.array([jnp.sum(self.norm(agent_location[agent] - agent_location)) for agent in range(self.num_drones)])
            * 0.05
            / (self.num_drones - 1)
            # a maximum value minus the distance to the target
            + 0.95 * (2 * self.size - self.norm(agent_location - self._target_location))
        )
        # negative reward if the drones crash
        +crash * -10 * jnp.ones(self.num_drones)

        """
        rewards = end * (
                rewards * 0.05 / (self.num_drones - 1)
                + 0.95 * (2 * self.size - self.norm(self._agent_location - self._target_location))
        ) + crash * -10 * jnp.ones(self.num_drones)
        """

        return rewards

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, timestep, crash, end, agent_location):
        # End of the game
        end = timestep == 100

        # collision with the ground and the target
        terminated = jnp.logical_or(agent_location[:, 2] < 0.2, jnp.linalg.norm(agent_location - self._target_location) < 0.2)

        for agent in range(self.num_drones):
            distances = self.norm(agent_location[agent] - agent_location)

            # collision between two drones
            terminated = terminated.at[agent].set(
                jnp.logical_or(terminated[agent], jnp.any(jnp.logical_and(distances > 0.001, distances < 0.2)))
            )

        crash = jnp.any(terminated)

        terminated = (crash + end) * jnp.ones(self.num_drones)

        return terminated, crash, end

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, timestep, end):
        end = timestep == 200

        truncation = end * jnp.ones(self.num_drones)

        return truncation, end

    @override
    # @partial(jit, static_argnums=(0,))
    def _compute_info(self):
        info = jnp.array([])
        return info


if __name__ == "__main__":
    parallel_env = Surround(
        num_drones=5,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
    )

    observations = parallel_env.reset()

    global_step = 0
    start_time = time.time()
    for i in range(100):
        while not parallel_env.end and not parallel_env.crash:
            actions = jnp.array([parallel_env.action_space().sample() for _ in range(parallel_env.num_drones)])
            # this is where you would insert your policy
            observations, rewards, terminations, truncations, infos = parallel_env.step(actions, parallel_env._mode)
            parallel_env.render()

            # print("obs", observations, "reward", rewards)
            # print("reward", rewards)

            if global_step % 100 == 0:
                print("SPS:", int(global_step / (time.time() - start_time)))

            global_step += 1

            # time.sleep(0.02)

        parallel_env.nb_crash += parallel_env.crash
        parallel_env.nb_end += parallel_env.end

        observations = parallel_env.reset()

    print("nb_crash", parallel_env.nb_crash)
    print("nb_end", parallel_env.nb_end)
