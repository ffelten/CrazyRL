"""Surround environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point."""
import time
from functools import partial
from typing_extensions import override
from dataclasses import dataclass

import jax_dataclasses as jdc
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit, vmap

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv


@override
@jdc.pytree_dataclass
class State:
    """State of the environment containing the modifiable variables."""
    agent_location: jnp.ndarray
    timestep: int
    crash: bool
    end: bool
    terminations: jnp.ndarray = jnp.array([])
    rewards: jnp.ndarray = jnp.array([])
    observations: jnp.ndarray = jnp.array([])
    infos: jnp.ndarray = jnp.array([])
    truncations: jnp.ndarray = jnp.array([])


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
        self.state = State(jnp.copy(init_flying_pos), 0, False, False)

        # Constant variables
        self.num_drones = num_drones

        self._target_location = target_location  # unique target location for all agents

        self._init_flying_pos = init_flying_pos

        self.size = size

        self.norm = vmap(jnp.linalg.norm)  # function to compute the norm of each array in a matrix

        super().__init__(
            render_mode=render_mode,
            size=size,
            init_flying_pos=self._init_flying_pos,
            target_location=self._target_location,
            num_drones=self.num_drones,
            swarm=swarm,
            state=self.state,
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
    def _compute_obs(self, state):
        obs = jnp.array([[] for _ in range(self.num_drones)])

        for agent in range(self.num_drones):
            obs_agent = jnp.append(state.agent_location[agent], self._target_location)

            for other_agent in range(self.num_drones):
                if other_agent != agent:
                    obs_agent = jnp.append(obs_agent, state.agent_location[other_agent])

            obs = obs.at[agent].set(obs_agent)

        return jdc.replace(state, observations=obs)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_action(self, actions, location):
        # Actions are clipped to stay in the map and scaled to do max 20cm in one step
        target_point_action = jnp.clip(location + actions * 0.2, jnp.array([-self.size, -self.size, 0]), self.size)

        return target_point_action

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state):
        # Reward is the mean distance to the other agents plus a maximum value minus the distance to the target

        rewards = state.end * (
            # mean distance to the other agents
            jnp.array([jnp.sum(self.norm(state.agent_location[agent] - state.agent_location)) for agent in range(self.num_drones)])
            * 0.05
            / (self.num_drones - 1)
            # a maximum value minus the distance to the target
            + 0.95 * (2 * self.size - self.norm(state.agent_location - self._target_location))
        )
        # negative reward if the drones crash
        + state.crash * -10 * jnp.ones(self.num_drones)

        return jdc.replace(state, rewards=rewards)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state):
        # End of the game
        end = state.timestep >= 100

        # collision with the ground and the target
        terminated = jnp.logical_or(state.agent_location[:, 2] < 0.2, jnp.linalg.norm(state.agent_location - self._target_location) < 0.2)

        for agent in range(self.num_drones):
            distances = self.norm(state.agent_location[agent] - state.agent_location)

            # collision between two drones
            terminated = terminated.at[agent].set(
                jnp.logical_or(terminated[agent], jnp.any(jnp.logical_and(distances > 0.001, distances < 0.2)))
            )

        crash = (end - 1) * jnp.any(terminated)

        terminated = (crash + end) * jnp.ones(self.num_drones)

        return jdc.replace(state, crash=crash, end=end, terminations=terminated)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state):
        end = state.end + (state.timestep == 200)

        truncations = end * jnp.ones(self.num_drones)

        return jdc.replace(state, end=end, truncations=truncations)

    @override
    def _compute_info(self, state):
        info = jnp.array([])
        return jdc.replace(state, infos=info)


if __name__ == "__main__":
    parallel_env = Surround(
        num_drones=5,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
    )

    parallel_env.state = parallel_env.reset(parallel_env.state)

    global_step = 0
    start_time = time.time()
    premier = True
    for i in range(500):
        while not parallel_env.state.end and not parallel_env.state.crash:
            actions = jnp.array([parallel_env.action_space().sample() for _ in range(parallel_env.num_drones)])
            # this is where you would insert your policy
            parallel_env.state = parallel_env.step(parallel_env.state, actions)

            # parallel_env.render()

            # print("obs", parallel_env.state.observations, "reward", parallel_env.state.rewards)
            # print("reward", parallel_env.state.rewards)

            if global_step % 2000 == 0:
                print("SPS:", int(global_step / (time.time() - start_time)))

            global_step += 1

            # time.sleep(0.02)

        if parallel_env.state.crash != 0:
            parallel_env.nb_crash += 1
        parallel_env.nb_end += parallel_env.state.end

        parallel_env.state = parallel_env.reset(parallel_env.state)

    print("nb_crash", parallel_env.nb_crash)
    print("nb_end", parallel_env.nb_end)
    print("total", parallel_env.nb_end + parallel_env.nb_crash)
