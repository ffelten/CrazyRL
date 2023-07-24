"""Escort environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point moving to one point to another."""

from functools import partial
from typing import Tuple
from typing_extensions import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import jit, random

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State
from crazy_rl.utils.jax_spaces import Box, Space
from crazy_rl.utils.jax_wrappers import (
    AddIDToObs,
    AutoReset,
    LogWrapper,
    NormalizeObservation,
    NormalizeVecReward,
    VecEnv,
)


@jdc.pytree_dataclass
class State(State):
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game
    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the common target


class Escort(BaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target going straight to one point to another."""

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        init_target_location: jnp.ndarray,
        final_target_location: jnp.ndarray,
        num_intermediate_points: int = 10,
        size: int = 3,
    ):
        """Escort environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            init_target_location: Array of the initial position of the moving target
            final_target_location: Array of the final position of the moving target
            num_intermediate_points: Number of intermediate points in the target trajectory
            size: Size of the map in meters
        """
        self.num_drones = num_drones

        self._target_location = init_target_location  # unique target location for all agents

        self._init_flying_pos = init_flying_pos

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
            high=self.size,
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
        finished = state.timestep < self.num_ref_points
        return jdc.replace(
            state,
            agents_locations=self._sanitize_action(state, actions),
            target_location=jnp.array([finished * self.ref[state.timestep] + (1 - finished) * self.ref[-1]]),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state: State, terminations: jnp.ndarray, truncations: jnp.ndarray) -> jnp.ndarray:
        # Reward is the mean distance to the other agents plus a maximum value minus the distance to the target

        reward_close_to_target = 0.95 * (
            2 * self.size - jnp.linalg.norm(state.agents_locations - state.target_location, axis=1)
        )
        reward_far_from_other_agents = (
            jnp.array(
                [
                    jnp.sum(jnp.linalg.norm(state.agents_locations[agent] - state.agents_locations, axis=1))
                    for agent in range(self.num_drones)
                ]
            )
            * 0.05
            / (self.num_drones - 1)
        )
        reward_crash = jnp.any(terminations) * -10 * jnp.ones(self.num_drones)

        return (
            jnp.any(truncations) * (reward_close_to_target + reward_far_from_other_agents)
            + (1 - jnp.any(truncations)) * reward_crash
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state: State) -> jnp.ndarray:
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

        return jnp.any(terminated) * jnp.ones(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state: State) -> jnp.ndarray:
        return (state.timestep == 100) * jnp.ones(self.num_drones)

    @override
    @partial(jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, dict, State]:
        state = State(
            agents_locations=self._init_flying_pos,
            timestep=0,
            target_location=jnp.array([self._target_location]),
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
    env = NormalizeObservation(env)
    env = AddIDToObs(
        env, num_agents
    )  # concats the agent id as one hot encoded vector to the obs (easier for learning algorithms)
    env = LogWrapper(env)  # Add stuff in the info dictionary for logging in the learning algo
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods
    env = NormalizeVecReward(env, gamma=0.99)  # normalize the reward in [-1, 1]

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
