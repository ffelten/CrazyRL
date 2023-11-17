"""Escort environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point moving to one point to another."""

from functools import partial
from typing import Tuple
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import jit, random

from crazy_rl.multi_agent.jax.base_parallel_env import (
    CLOSENESS_THRESHOLD,
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
    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the common target
    prev_agent_locations: jnp.ndarray  # 2D array containing x,y,z coordinates of each agent at last timestep
    prev_target_locations: jnp.ndarray  # 2D array containing x,y,z coordinates of the target of each agent at last timestep


class Escort(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target going straight to one point to another."""

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        init_target_location: jnp.ndarray,
        final_target_location: jnp.ndarray,
        num_intermediate_points: int = 20,
        multi_obj: bool = False,
        size: int = 2,
    ):
        """Escort environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            init_target_location: Array of the initial position of the moving target
            final_target_location: Array of the final position of the moving target
            num_intermediate_points: Number of intermediate points in the target trajectory
            multi_obj: Whether to return a multi-objective reward
            size: Size of the map in meters
        """
        self.num_drones = num_drones

        self._target_location = init_target_location  # unique target location for all agents

        self._init_flying_pos = init_flying_pos
        self.multi_obj = multi_obj
        # There are two more ref points than intermediate points, one for the initial and final target locations
        self.num_ref_points = num_intermediate_points + 2
        # Ref is a 2d arrays for the target
        # it contains the reference points (xyz) for the target at each timestep

        self.ref = jnp.array(
            [
                init_target_location + ((final_target_location - init_target_location) * t / self.num_ref_points)
                for t in range(self.num_ref_points)
            ]
        )

        self.size = size

    @override
    def observation_space(self, agent: int) -> Space:
        return Box(
            low=-self.size,
            high=3,
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
        )

    @override
    def action_space(self, agent: int) -> Space:
        return Box(low=-1, high=1, shape=(3,))  # 3D speed vector

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state: State) -> jnp.ndarray:
        return jnp.append(
            # each row contains the location of one agent and the location of the target
            jnp.column_stack((state.agents_locations, jnp.tile(state.target_location, (self.num_drones, 1)))),
            # then we add agents_locations to each row without the agent which is already in the row
            # and make it only one dimension
            jnp.array([jnp.delete(state.agents_locations, agent, axis=0).flatten() for agent in range(self.num_drones)]),
            axis=1,
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _transition_state(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        not_finished = state.timestep < self.num_ref_points
        prev_agent_locations = state.agents_locations
        prev_target_locations = state.target_location
        return jdc.replace(
            state,
            agents_locations=self._sanitize_action(state, actions),
            target_location=jnp.array([not_finished * self.ref[state.timestep] + (1 - not_finished) * self.ref[-1]]),
            prev_agent_locations=prev_agent_locations,
            prev_target_locations=prev_target_locations,
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state: State, terminations: jnp.ndarray, truncations: jnp.ndarray) -> jnp.ndarray:
        # Potential based reward (!) locations and targets must be updated before this
        dist_from_old_target = _distances_to_target(state.agents_locations, state.prev_target_locations)
        old_dist = _distances_to_target(state.prev_agent_locations, state.prev_target_locations)
        # reward should be new_potential - old_potential but since the potential should be negated (we want to min distance),
        # we have to negate the reward, -new_potential - (-old_potential) = old_potential - new_potential
        reward_close_to_target = old_dist - dist_from_old_target

        reward_far_from_other_agents = jnp.array(
            [
                jnp.sum(jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1))
                for agent in range(self.num_drones)
            ]
        ) / (self.num_drones - 1)
        reward_crash = jnp.any(terminations) * -10 * jnp.ones(self.num_drones)

        if self.multi_obj:
            return (1 - jnp.any(terminations)) * jnp.column_stack(
                (reward_close_to_target, reward_far_from_other_agents)
            ) + jnp.any(terminations) * jnp.column_stack((reward_crash, reward_crash))
        else:
            # MO reward linearly combined using hardcoded weights
            return (1 - jnp.any(terminations)) * (
                0.995995 * reward_close_to_target + 0.004005 * reward_far_from_other_agents
            ) + jnp.any(terminations) * reward_crash

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state: State) -> jnp.ndarray:
        # collision with the ground and the target
        terminated = jnp.logical_or(
            state.agents_locations[:, 2] < CLOSENESS_THRESHOLD,
            jnp.linalg.norm(state.agents_locations - state.target_location) < CLOSENESS_THRESHOLD,
        )

        for agent in range(self.num_drones):
            distances = jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1)

            # collision between two drones
            terminated = terminated.at[agent].set(
                jnp.logical_or(terminated[agent], jnp.any(jnp.logical_and(distances > 0.001, distances < CLOSENESS_THRESHOLD)))
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
            target_location=jnp.array([self._target_location]),
            prev_target_locations=jnp.array([self._target_location]),
        )
        obs = self._compute_obs(state)
        return obs, {}, state

    @override
    @partial(jit, static_argnums=(0,))
    def state(self, state: State) -> jnp.ndarray:
        return jnp.append(state.agents_locations.flatten(), state.target_location)


if __name__ == "__main__":
    from jax.lib import xla_bridge

    jax.config.update("jax_platform_name", "gpu")

    print(xla_bridge.get_backend().platform)

    num_agents = 5
    env = Escort(
        num_drones=num_agents,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        init_target_location=jnp.array([1.0, 1.0, 2.5]),
        final_target_location=jnp.array([-2.0, -2.0, 3.0]),
        num_intermediate_points=150,
    )

    num_envs = 1000  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, num_envs + 1)

    # Wrappers
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods

    obs, info, state = env.reset(jnp.stack(subkeys))

    for i in range(201):
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
