"""Surround environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point."""
from functools import partial
from typing import Tuple
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from gymnasium import spaces
from jax import jit, random

from crazy_rl.multi_agent.jax.base_parallel_env import (
    BaseParallelEnv,
    State,
    _distances_to_target,
)
from crazy_rl.utils.jax_spaces import Box, Space
from crazy_rl.utils.jax_wrappers import AutoReset, VecEnv


@jdc.pytree_dataclass
class State(State):
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game
    prev_agent_locations: jnp.ndarray  # 2D array containing x,y,z coordinates of each agent at last timestep


class Surround(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a target point."""

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        target_location: jnp.ndarray,
        multi_obj: bool = False,
        size: int = 3,
    ):
        """Surround environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            target_location: Array of the position of the target point
            multi_obj: Whether to return a multi-objective reward
            size: Size of the map in meters
        """
        self.num_drones = num_drones
        self._target_location = target_location  # unique target location for all agents
        self._init_flying_pos = init_flying_pos
        self.multi_obj = multi_obj
        self.size = size

    @override
    def observation_space(self, agent: int) -> Space:
        return spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
            dtype=jnp.float32,
        )

    @override
    def action_space(self, agent: int) -> Space:
        return Box(low=-1, high=1, shape=(3,))  # 3D speed vector

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state: State) -> jnp.ndarray:
        return jnp.append(
            # each row contains the location of one agent and the location of the target
            jnp.column_stack((state.agents_locations, jnp.tile(self._target_location, (self.num_drones, 1)))),
            # then we add agents_locations to each row without the agent which is already in the row
            # and make it only one dimension
            jnp.array([jnp.delete(state.agents_locations, agent, axis=0).flatten() for agent in range(self.num_drones)]),
            axis=1,
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _transition_state(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        return jdc.replace(
            state, agents_locations=self._sanitize_action(state, actions), prev_agent_locations=state.agents_locations
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state: State, terminations: jnp.ndarray, truncations: jnp.ndarray) -> jnp.ndarray:
        # REWARD DISTANCE FROM OTHERS
        # Reward is the mean distance to the other agents plus a maximum value minus the distance to the target
        reward_far_from_other_agents = jnp.array(
            [
                jnp.sum(jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1))
                for agent in range(self.num_drones)
            ]
        ) / (self.num_drones - 1)

        # REWARD DISTANCE FROM TARGET
        cur_dist_to_target = _distances_to_target(state.agents_locations, self._target_location)
        old_dist = _distances_to_target(state.prev_agent_locations, self._target_location)
        # reward should be new_potential - old_potential but since the potential should be negated (we want to min distance),
        # we have to negate the reward, -new_potential - (-old_potential) = old_potential - new_potential
        reward_close_to_target = old_dist - cur_dist_to_target

        reward_crash = -10 * jnp.ones(self.num_drones)

        if self.multi_obj:
            return (1 - jnp.any(terminations)) * jnp.column_stack(
                (reward_close_to_target, reward_far_from_other_agents)
            ) + jnp.any(terminations) * jnp.column_stack((reward_crash, reward_crash))
        else:
            # MO reward linearly combined using hardcoded weights
            return (1 - jnp.any(terminations)) * (
                0.995 * reward_close_to_target + 0.005 * reward_far_from_other_agents
            ) + jnp.any(terminations) * reward_crash

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state: State) -> jnp.ndarray:
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

        return jnp.any(terminated) * jnp.ones(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state: State) -> jnp.ndarray:
        return (state.timestep == 200) * jnp.ones(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, dict, State]:
        state = State(
            agents_locations=self._init_flying_pos,
            prev_agent_locations=self._init_flying_pos,
            timestep=0,
        )
        obs = self._compute_obs(state)
        return obs, {}, state

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state: State) -> jnp.ndarray:
        return jnp.append(state.agents_locations.flatten(), self._target_location)


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    num_agents = 5
    env = Surround(
        num_drones=num_agents,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        target_location=jnp.array([[1.0, 1.0, 2.5]]),
    )

    num_envs = 1000  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, num_envs + 1)

    # Wrappers
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods

    obs, info, state = env.reset(jnp.stack(subkeys))

    for i in range(301):
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

        # print("obs", obs)
        print("rewards", rewards)
        # print("term", term)
        print("trunc", trunc)
        # print("info", info)
