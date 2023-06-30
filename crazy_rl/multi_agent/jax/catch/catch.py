"""Catch environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point trying to escape."""

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

    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the common target


class Catch(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target trying to escape."""

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        init_target_location: jnp.ndarray,
        target_speed: float,
        size: int = 3,
    ):
        """Catch environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            init_target_location: Array of the initial position of the moving target
            target_speed: Distance traveled by the target at each timestep
            size: Size of the map in meters
        """
        self.num_drones = num_drones

        self._target_location = init_target_location  # unique target location for all agents

        self.target_speed = target_speed

        self._init_flying_pos = init_flying_pos

        self.size = size

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
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state):
        return jdc.replace(
            state,
            observations=jnp.append(
                # each row contains the location of one agent and the location of the target
                jnp.column_stack((state.agents_locations, jnp.tile(state.target_location, (self.num_drones, 1)))),
                # then we add agents_locations to each row without the agent which is already in the row
                # and make it only one dimension
                jnp.array([jnp.delete(state.agents_locations, agent, axis=0).flatten() for agent in range(self.num_drones)]),
                axis=1,
            ),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _transition_state(self, state, actions, key):
        state = self._sanitize_action(state, actions)

        # Mechanics of the target

        # mean of the agent's positions
        mean = jnp.zeros(3)

        for agent in range(self.num_drones):
            mean += state.agents_locations[agent]

        mean /= self.num_drones

        dist = jnp.linalg.norm(mean - state.target_location[0])

        surrounded = dist <= 0.2

        # if the target is out of the map, put it back in the map
        target_location = jnp.clip(
            state.target_location
            # go to the opposite direction of the mean of the agents
            + ((1 - surrounded) * (state.target_location - mean) / (dist + 0.0001) * self.target_speed)
            # if the mean of the agents is too close to the target, move the target in a random direction,
            # slowly because it hesitates
            + (surrounded * random.uniform(key, (3,), minval=-1, maxval=1) * self.target_speed * 0.1),
            jnp.array([-self.size, -self.size, 0.2]),
            jnp.array([self.size, self.size, self.size]),
        )

        return jdc.replace(state, target_location=target_location)

    @override
    @partial(jit, static_argnums=(0,))
    def _sanitize_action(self, state, actions):
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
                        jnp.sum(jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1))
                        for agent in range(self.num_drones)
                    ]
                )
                * 0.05
                / (self.num_drones - 1)
                # a maximum value minus the distance to the target
                + 0.95 * (2 * self.size - jnp.linalg.norm(state.agents_locations - state.target_location, axis=1))
            )
            # negative reward if the drones crash
            + jnp.any(state.terminations) * (1 - jnp.any(state.truncations)) * -10 * jnp.ones(self.num_drones),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state):
        # collision with the ground and the target
        terminated = jnp.logical_or(
            state.agents_locations[:, 2] < 0.2, jnp.linalg.norm(state.agents_locations - state.target_location) < 0.2
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
    def _compute_truncation(self, state):
        return jdc.replace(state, truncations=(state.timestep == 100) * jnp.ones(self.num_drones))

    @override
    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        """Resets the environment in initial state."""
        state = State(
            agents_locations=self._init_flying_pos,
            timestep=0,
            observations=jnp.array([]),
            rewards=jnp.zeros(self.num_drones),
            terminations=jnp.zeros(self.num_drones),
            truncations=jnp.zeros(self.num_drones),
            target_location=jnp.array([self._target_location]),
        )
        state = self._compute_obs(state)
        return state

    @override
    @partial(jit, static_argnums=(0,))
    def auto_reset(self, **state):
        """Returns the State reinitialized if needed, else the actual State.

        The values contained by State are passed in argument and used like a dictionary
        because auto_reset is meant to be used by vmap and vmap doesn't accept objects.
        """
        done = jnp.any(state["truncations"]) + jnp.any(state["terminations"])

        state = State(
            agents_locations=done * self._init_flying_pos + (1 - done) * state["agents_locations"],
            timestep=(1 - done) * state["timestep"],
            observations=state["observations"],
            rewards=(1 - done) * state["rewards"],
            terminations=(1 - done) * state["terminations"],
            truncations=(1 - done) * state["truncations"],
            target_location=done * self._target_location + (1 - done) * state["target_location"],
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
            "target_location": state.target_location,
        }

    @partial(jit, static_argnums=(0,))
    def step_vmap(self, action, key, **state_val):
        """Used to vmap step, takes the values of the state and calls step with a new State object containing the state values.

        Args:
            action: 2D array containing the x, y, z action for each drone.
            key : JAX PRNG key.
            **state_val: Different values contained in the State.
        """
        return self.step(State(**state_val), action, key)

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state):
        """Returns a global observation (concatenation of all the agent locations and target locations)."""
        return jnp.append(state.agents_locations.flatten(), state.target_location)


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    parallel_env = Catch(
        num_drones=5,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        init_target_location=jnp.array([1.0, 1.0, 2.5]),
        target_speed=0.1,
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
