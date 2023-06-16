"""Surround environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point."""
import time
from functools import partial
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from gymnasium import spaces
from jax import jit, random, vmap

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv


@override
@jdc.pytree_dataclass
class State:
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game

    observations: jnp.ndarray  # array containing the current observation of each agent
    rewards: jnp.ndarray  # array containing the current reward of each agent
    terminations: jnp.ndarray  # array of booleans which are True if the agents have crashed
    truncations: jnp.ndarray  # array of booleans which are True if the game reaches 100 timesteps

    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the unique target


class Surround(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a target point."""

    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        target_location: jnp.ndarray,
        size: int = 3,
    ):
        """Surround environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            target_location: Array of the position of the target point
            size: Size of the map
        """
        self.num_drones = num_drones

        self._target_location = target_location  # unique target location for all agents

        self._init_flying_pos = init_flying_pos

        self.size = size

        self.norm = vmap(jnp.linalg.norm)  # function to compute the norm of each array in a matrix

        super().__init__(
            size=size,
            init_flying_pos=self._init_flying_pos,
            target_location=self._target_location,
            num_drones=self.num_drones,
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
    def _compute_obs(self, state, key):
        return jdc.replace(
            state,
            observations=jnp.append(
                jnp.column_stack((state.agents_locations, jnp.tile(state.target_location, (self.num_drones, 1)))),
                jnp.array([jnp.delete(state.agents_locations, agent, axis=0).flatten() for agent in range(self.num_drones)]),
                axis=1,
            ),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_action(self, state, actions):
        # Actions are clipped to stay in the map and scaled to do max 20cm in one step
        return jdc.replace(
            state,
            agents_locations=jnp.clip(
                state.agents_locations + actions * 0.2, jnp.array([-self.size, -self.size, 0]), self.size
            ),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state):
        # Reward is the mean distance to the other agents plus a maximum value minus the distance to the target

        return jdc.replace(
            state,
            rewards=jnp.any(state.truncations)
            * (
                # mean distance to the other agents
                jnp.array(
                    [
                        jnp.sum(self.norm(state.agents_locations[agent] - state.agents_locations))
                        for agent in range(self.num_drones)
                    ]
                )
                * 0.05
                / (self.num_drones - 1)
                # a maximum value minus the distance to the target
                + 0.95 * (2 * self.size - self.norm(state.agents_locations - state.target_location))
            )
            # negative reward if the drones crash
            + jnp.any(state.terminations) * -10 * jnp.ones(self.num_drones),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state):
        # collision with the ground and the target
        terminated = jnp.logical_or(
            state.agents_locations[:, 2] < 0.2, self.norm(state.agents_locations - state.target_location) < 0.2
        )

        for agent in range(self.num_drones):
            distances = self.norm(state.agents_locations[agent] - state.agents_locations)

            # collision between two drones
            terminated = terminated.at[agent].set(
                jnp.logical_or(terminated[agent], jnp.any(jnp.logical_and(distances > 0.001, distances < 0.2)))
            )

        return jdc.replace(
            state, terminations=((1 - jnp.any(state.truncations)) * jnp.any(terminated)) * jnp.ones(self.num_drones)
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state):
        return jdc.replace(state, truncations=(state.timestep == 100) * jnp.ones(self.num_drones))

    @override
    @partial(jit, static_argnums=(0,))
    def _initialize_state(self):
        return State(
            self._init_flying_pos,
            0,
            jnp.array([]),
            jnp.array([]),
            jnp.zeros(self.num_drones),
            jnp.zeros(self.num_drones),
            self._target_location,
        )

    @override
    @partial(jit, static_argnums=(0,))
    def auto_reset(self, **state):
        """Resets if the game has ended, or returns state."""
        done = jnp.any(state["truncations"]) + jnp.any(state["terminations"])

        return State(
            done * self._init_flying_pos + (1 - done) * state["agents_locations"],
            (1 - done) * state["timestep"],
            (1 - done) * state["observations"],
            (1 - done) * state["rewards"],
            (1 - done) * state["terminations"],
            (1 - done) * state["truncations"],
            done * self._target_location + (1 - done) * state["target_location"],
        )

    @partial(jit, static_argnums=(0,))
    def state_to_dict(self, state):
        """Translates the State into a dict."""
        return {
            "agents_locations": state.agents_locations,
            "timestep": state.timestep,
            "observations": state.observations,
            "rewards": state.rewards,
            "terminations": state.terminations,
            "truncations": state.truncations,
            "target_location": state.target_location,
        }

    @partial(jit, static_argnums=(0,))
    def step_vmap(self, action, key, **state_val):
        """Calls step with a State and is called by vmap without State object."""
        return self.step(State(**state_val), action, key)


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "cpu")

    print(xla_bridge.get_backend().platform)

    parallel_env = Surround(
        num_drones=5,
        init_flying_pos=jnp.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
    )

    n = 200  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)

    @jit
    def play(key):
        """Execution of the environment with random actions."""
        key, *subkeys = random.split(key, n + 1)

        states = vmap(parallel_env.reset)(jnp.stack(subkeys))

        for i in range(500):
            actions = random.uniform(key, (n, parallel_env.num_drones, 3), minval=-1, maxval=1)

            key, *subkeys = random.split(key, n + 1)

            states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))

            states = vmap(parallel_env.auto_reset)(**parallel_env.state_to_dict(states))

        return key

    key = play(key)  # compilation of the function

    start = time.time()

    play(key)

    print("total duration :", time.time() - start)

    # with profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
