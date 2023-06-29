"""Hover environment for Crazyflies 2."""
import time
from functools import partial
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from gymnasium import spaces
from jax import jit, random, vmap

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State


@jdc.pytree_dataclass
class State(State):
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game

    observations: jnp.ndarray  # array containing the current observation of each agent
    rewards: jnp.ndarray  # array containing the current reward of each agent
    terminations: jnp.ndarray  # array of booleans which are True if the agents have crashed
    truncations: jnp.ndarray  # array of booleans which are True if the game reaches 100 timesteps


class Hover(BaseParallelEnv):
    """A Parallel Environment where drone learn how to hover around a target point."""

    metadata = {"is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        size: int = 3,
    ):
        """Hover environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            size: Size of the map
        """
        self.num_drones = num_drones

        self._init_flying_pos = init_flying_pos
        self._target_location = init_flying_pos

        self.size = size

        super().__init__()

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=np.array([-self.size, -self.size, 0, -self.size, -self.size, 0], dtype=np.float32),
            high=np.array([self.size, self.size, self.size, self.size, self.size, self.size], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

    @override
    def _action_space(self):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state):
        return jdc.replace(state, observations=vmap(jnp.append)(state.agents_locations, self._target_location))

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state):
        return jnp.append(state.agents_locations, self._target_location).flatten()

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_mechanics(self, state, key):
        return state

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_action(self, state, actions):
        return jdc.replace(
            state,
            agents_locations=jnp.clip(
                state.agents_locations + actions * 0.2, jnp.array([-self.size, -self.size, 0]), self.size
            ),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state):
        return jdc.replace(state, rewards=-1 * jnp.linalg.norm(self._target_location - state.agents_locations, axis=1))

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state):
        # the drones never crash. Terminations initialized to jnp.zeros() and then never changes
        return state

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state):
        return jdc.replace(state, truncations=(state.timestep == 100) * jnp.ones(self.num_drones))

    @override
    @partial(jit, static_argnums=(0,))
    def _initialize_state(self):
        return State(
            agents_locations=self._init_flying_pos,
            timestep=0,
            observations=jnp.array([]),
            rewards=jnp.zeros(self.num_drones),
            terminations=jnp.zeros(self.num_drones),
            truncations=jnp.zeros(self.num_drones),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def auto_reset(self, **state):
        """Resets if the game has ended, or returns state."""
        done = jnp.any(state["truncations"]) + jnp.any(state["terminations"])

        state = State(
            agents_locations=done * self._init_flying_pos + (1 - done) * state["agents_locations"],
            timestep=(1 - done) * state["timestep"],
            observations=state["observations"],
            rewards=(1 - done) * state["rewards"],
            terminations=(1 - done) * state["terminations"],
            truncations=(1 - done) * state["truncations"],
        )
        state = self._compute_obs(state)
        return state

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
        }

    @partial(jit, static_argnums=(0,))
    def step_vmap(self, action, key, **state_val):
        """Calls step with a State and is called by vmap without State object."""
        return self.step(State(**state_val), action, key)


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    parallel_env = Hover(
        num_drones=5,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
    )

    n = 1000  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)

    @jit
    def body(i, states_key):
        """Body of the fori_loop of play."""
        actions = random.uniform(states_key[1], (n, parallel_env.num_drones, 3), minval=-1, maxval=1)

        key, *subkeys = random.split(states_key[1], n + 1)

        states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states_key[0]))

        states = vmap(parallel_env.auto_reset)(**parallel_env.state_to_dict(states))

        return (states, key)

    @jit
    def play(key):
        """Execution of the environment with random actions."""
        key, *subkeys = random.split(key, n + 1)

        states = vmap(parallel_env.reset)(jnp.stack(subkeys))

        states, key = jax.lax.fori_loop(0, 1000, body, (states, key))

        return key, states

    key, states = play(key)  # compilation of the function

    durations = np.zeros(10)

    print("start")

    for i in range(10):
        start = time.time()

        key, states = play(key)

        jax.block_until_ready(states)

        end = time.time() - start

        durations[i] = end

    print("durations : ", durations)
