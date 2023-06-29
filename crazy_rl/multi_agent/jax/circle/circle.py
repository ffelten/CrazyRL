"""Circle environment for Crazyflie 2. Each agent is supposed to learn to perform a circle around a target point."""
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

    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the target of each agent


class Circle(BaseParallelEnv):
    """A Parallel Environment where drone learn how to perform a circle."""

    metadata = {"is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        num_intermediate_points: int = 10,
        size: int = 3,
    ):
        """Circle environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            num_intermediate_points: Number of intermediate points in the target circle
            size: Size of the map
        """
        self.num_drones = num_drones

        self.size = size

        self._init_flying_pos = init_flying_pos

        # Specific to circle

        circle_radius = 0.5  # [m]

        self.num_intermediate_points = num_intermediate_points

        # Ref is a list of 2d arrays for each agent
        # each 2d array contains the reference points (xyz) for the agent at each timestep
        self.ref = jnp.zeros((num_intermediate_points, self.num_drones, 3))

        ts = 2 * jnp.pi * jnp.arange(num_intermediate_points) / num_intermediate_points

        for agent in range(self.num_drones):
            self.ref = self.ref.at[:, agent, 0].set(
                circle_radius * (1 - jnp.cos(ts)) + (init_flying_pos[agent][0] - circle_radius)
            )
            self.ref = self.ref.at[:, agent, 1].set(circle_radius * jnp.sin(ts) + (init_flying_pos[agent][1]))
            self.ref = self.ref.at[:, agent, 2].set(init_flying_pos[agent][2])

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=jnp.array([-self.size, -self.size, 0, -self.size, -self.size, 0], dtype=jnp.float32),
            high=jnp.array([self.size, self.size, self.size, self.size, self.size, self.size], dtype=jnp.float32),
            shape=(6,),
            dtype=jnp.float32,
        )

    @override
    def _action_space(self):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state):
        return jdc.replace(state, observations=vmap(jnp.append)(state.agents_locations, state.target_location))

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_mechanics(self, state, key):
        return jdc.replace(
            state, target_location=self.ref[state.timestep % self.num_intermediate_points]
        )  # redo the circle if the end is reached

    @override
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
        # Reward is based on the Euclidean distance to the target point

        return jdc.replace(state, rewards=-1 * jnp.linalg.norm(state.target_location - state.agents_locations, axis=1))

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
            target_location=jnp.copy(self.ref[0]),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def auto_reset(self, **state):
        """Returns the State reinitialized if needed, else the actual State."""
        done = jnp.any(state["truncations"]) + jnp.any(state["terminations"])

        state = State(
            agents_locations=done * self._init_flying_pos + (1 - done) * state["agents_locations"],
            timestep=(1 - done) * state["timestep"],
            observations=state["observations"],
            rewards=(1 - done) * state["rewards"],
            terminations=(1 - done) * state["terminations"],
            truncations=(1 - done) * state["truncations"],
            target_location=done * self.ref[0] + (1 - done) * state["target_location"],
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
        """Calls step with a State and is called by vmap without State object."""
        return self.step(State(**state_val), action, key)

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state):
        """Returns a global observation (concatenation of all the agent locations and target locations)."""
        return jnp.append(state.agents_locations, state.target_location).flatten()


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    parallel_env = Circle(
        num_drones=5,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        num_intermediate_points=100,
    )
    n = 1000  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)

    @jit
    def body(i, states_key):
        """Body of the fori_loop of play.

        Args:
            i: number of the iteration.
            states_key: a tuple containing states and key.
        """
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
