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


class Surround(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a target point."""

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
            size: Size of the map in meters
        """
        self.num_drones = num_drones

        self._target_location = target_location  # unique target location for all agents

        self._init_flying_pos = init_flying_pos

        self.size = size

    @override
    def observation_space(self, agent: int) -> spaces.Space:
        return spaces.Box(
            low=np.tile(np.array([-self.size, -self.size, 0], dtype=np.float32), self.num_drones + 1),
            high=np.tile(np.array([self.size, self.size, self.size], dtype=np.float32), self.num_drones + 1),
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
            dtype=jnp.float32,
        )

    @override
    def action_space(self, agent: int) -> spaces.Space:
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state: State) -> State:
        return jdc.replace(
            state,
            observations=jnp.append(
                # each row contains the location of one agent and the location of the target
                jnp.column_stack((state.agents_locations, jnp.tile(self._target_location, (self.num_drones, 1)))),
                # then we add agents_locations to each row without the agent which is already in the row
                # and make it only one dimension
                jnp.array([jnp.delete(state.agents_locations, agent, axis=0).flatten() for agent in range(self.num_drones)]),
                axis=1,
            ),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _transition_state(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        return jdc.replace(state, agents_locations=self._sanitize_action(state, actions))

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state: State) -> State:
        # Reward is the mean distance to the other agents plus a maximum value minus the distance to the target

        return jdc.replace(
            state,
            rewards=jnp.any(state.truncations)
            * (
                jnp.array(
                    [  # mean distance to the other agents
                        jnp.sum(jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1))
                        for agent in range(self.num_drones)
                    ]
                )
                * 0.05
                / (self.num_drones - 1)
                # a maximum value minus the distance to the target
                + 0.95 * (2 * self.size - jnp.linalg.norm(state.agents_locations - self._target_location, axis=1))
            )
            # negative reward if the drones crash
            + jnp.any(state.terminations) * (1 - jnp.any(state.truncations)) * -10 * jnp.ones(self.num_drones),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state: State) -> State:
        # collision with the ground and the target
        terminated = jnp.logical_or(
            state.agents_locations[:, 2] < 0.2, jnp.linalg.norm(state.agents_locations - self._target_location, axis=1) < 0.2
        )

        for agent in range(self.num_drones):
            distances = jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1)

            # collision between two drones
            terminated = terminated.at[agent].set(
                jnp.logical_or(terminated[agent], jnp.any(jnp.logical_and(distances > 0.001, distances < 0.2)))
            )

        return jdc.replace(state, terminations=jnp.any(terminated) * jnp.ones(self.num_drones))

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state: State) -> State:
        return jdc.replace(state, truncations=(state.timestep == 100) * jnp.ones(self.num_drones))

    @override
    @partial(jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray) -> State:
        state = State(
            agents_locations=self._init_flying_pos,
            timestep=0,
            observations=jnp.array([]),
            rewards=jnp.zeros(self.num_drones),
            terminations=jnp.zeros(self.num_drones),
            truncations=jnp.zeros(self.num_drones),
        )
        state = self._compute_obs(state)
        return state

    @override
    @partial(jit, static_argnums=(0,))
    def auto_reset(self, **state) -> State:
        done = jnp.min(jnp.array([jnp.any(state["truncations"]) + jnp.any(state["terminations"]), 1]))

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

    @override
    @partial(jit, static_argnums=(0,))
    def state_to_dict(self, state: State) -> dict:
        return {
            "agents_locations": state.agents_locations,
            "timestep": state.timestep,
            "observations": state.observations,
            "rewards": state.rewards,
            "terminations": state.terminations,
            "truncations": state.truncations,
        }

    @override
    @partial(jit, static_argnums=(0,))
    def step_vmap(self, action: jnp.ndarray, key: jnp.ndarray, **state_val) -> State:
        return self.step(State(**state_val), action, key)

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state: State) -> jnp.ndarray:
        return jnp.append(state.agents_locations.flatten(), self._target_location)


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    parallel_env = Surround(
        num_drones=5,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        target_location=jnp.array([[1.0, 1.0, 2.5]]),
    )

    num_envs = 1000  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)

    vmapped_step = vmap(parallel_env.step_vmap)
    vmapped_auto_reset = vmap(parallel_env.auto_reset)
    vmapped_reset = vmap(parallel_env.reset)

    @jit
    def body(i, states_key):
        """Body of the fori_loop of play.

        Args:
            i: number of the iteration.
            states_key: a tuple containing states and key.
        """
        states, key = states_key

        actions = random.uniform(key, (num_envs, parallel_env.num_drones, 3), minval=-1, maxval=1)

        key, *subkeys = random.split(key, num_envs + 1)

        states = vmapped_step(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))

        # where you would learn or add to buffer

        states = vmapped_auto_reset(**parallel_env.state_to_dict(states))

        return (states, key)

    @jit
    def play(key):
        """Execution of the environment with random actions."""
        key, *subkeys = random.split(key, num_envs + 1)

        states = vmapped_reset(jnp.stack(subkeys))

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
