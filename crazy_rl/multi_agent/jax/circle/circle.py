"""Circle environment for Crazyflie 2. Each agent is supposed to learn to perform a circle around a target point."""
from functools import partial
from typing import Tuple
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import jit, random, vmap

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State
from crazy_rl.utils.jax_spaces import Box, Space
from crazy_rl.utils.jax_wrappers import AutoReset, LogWrapper, VecEnv


@jdc.pytree_dataclass
class State(State):
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game
    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the target of each agent


class Circle(BaseParallelEnv):
    """A Parallel Environment where drone learn how to perform a circle."""

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
            size: Size of the map in meters
        """
        self.num_drones = num_drones
        self.size = size
        self._init_flying_pos = init_flying_pos

        # Specific to circle
        circle_radius = 0.5  # [m]
        self.num_intermediate_points = num_intermediate_points

        # Ref is a list of 2d arrays for each agent
        # each 2d array contains the reference points (xyz) for the agent at each timestep
        self.ref = jnp.zeros((self.num_drones, num_intermediate_points, 3))
        ts = 2 * jnp.pi * jnp.arange(num_intermediate_points) / num_intermediate_points
        for agent in range(self.num_drones):
            self.ref = self.ref.at[agent, :, 0].set(
                circle_radius * (1 - jnp.cos(ts)) + (init_flying_pos[agent][0] - circle_radius)
            )
            self.ref = self.ref.at[agent, :, 1].set(circle_radius * jnp.sin(ts) + (init_flying_pos[agent][1]))
            self.ref = self.ref.at[agent, :, 2].set(init_flying_pos[agent][2])

    @override
    def observation_space(self, agent: int) -> Space:
        return Box(
            low=-self.size,
            high=self.size,
            shape=(6,),
        )

    @override
    def action_space(self, agent: int) -> Space:
        return Box(low=-1, high=1, shape=(3,))  # 3D speed vector

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state: State) -> jnp.ndarray:
        return vmap(jnp.append)(state.agents_locations, state.target_location)

    @override
    @partial(jit, static_argnums=(0,))
    def _transition_state(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        return jdc.replace(
            state,
            agents_locations=self._sanitize_action(state, actions),
            target_location=self.ref[:, state.timestep % self.num_intermediate_points, :],
        )  # redo the circle if the end is reached

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state: State, terminations: jnp.ndarray, truncations: jnp.ndarray) -> jnp.ndarray:
        # Reward is based on the Euclidean distance to the target point
        return 2 * self.size - jnp.linalg.norm(state.target_location - state.agents_locations, axis=1)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state: State) -> jnp.ndarray:
        # the drones never crash. Terminations initialized to jnp.zeros() and then never changes
        return jnp.zeros(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state: State) -> jnp.ndarray:
        return (state.timestep == 200) * jnp.ones(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, dict, State]:
        state = State(
            agents_locations=self._init_flying_pos,
            timestep=0,
            target_location=jnp.copy(self.ref[:, 0, :]),
        )
        obs = self._compute_obs(state)
        return obs, {}, state

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state: State) -> jnp.ndarray:
        return jnp.append(state.agents_locations, state.target_location).flatten()


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    num_agents = 3
    env = Circle(
        num_drones=num_agents,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        num_intermediate_points=10,
    )

    num_envs = 3  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, num_envs + 1)

    # Wrappers
    env = LogWrapper(env)  # Logs the env info
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods

    obs, info, state = env.reset(jnp.stack(subkeys))
    r = jnp.zeros(num_envs)

    for i in range(100):
        key, *subkeys = random.split(key, num_agents + 1)
        actions = (
            jnp.array([env.action_space(agent_id).sample(jnp.stack(subkeys[agent_id])) for agent_id in range(env.num_drones)])
            .flatten()
            .repeat(num_envs)
            .reshape((num_envs, env.num_drones, -1))
        )
        global_state = env.state(state)
        key, *subkeys = random.split(key, num_envs + 1)
        obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))
        r += rewards.sum(axis=1)

        # print("obs", obs)
        print("rewards", rewards)
        # print("term", term)
        # print("trunc", trunc)
        # print("info", info)
    print(r)
