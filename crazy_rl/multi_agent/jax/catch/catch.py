"""Catch environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point trying to escape."""

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
    NormalizeVecObservation,
    NormalizeVecReward,
    VecEnv,
)


@jdc.pytree_dataclass
class State(State):
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game
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
    def observation_space(self, agent: int) -> Space:
        return Box(
            low=-self.size,
            high=self.size,
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
        )

    @override
    def action_space(self, agent: int) -> Space:
        return Box(low=-1.0, high=1.0, shape=(3,))  # 3d speed vector of the drone

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
        agents_locations = self._sanitize_action(state, actions)

        # Mechanics of the target

        # mean of the agent's positions
        mean = jnp.zeros(3)

        for agent in range(self.num_drones):
            mean += agents_locations[agent]

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

        return jdc.replace(state, agents_locations=agents_locations, target_location=target_location)

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

    jax.config.update("jax_platform_name", "cpu")

    print(xla_bridge.get_backend().platform)

    num_agents = 5
    env = Catch(
        num_drones=num_agents,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        init_target_location=jnp.array([1.0, 1.0, 2.5]),
        target_speed=0.1,
    )

    num_envs = 3  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, num_envs + 1)

    # Wrappers
    env = AddIDToObs(
        env, num_agents
    )  # concats the agent id as one hot encoded vector to the obs (easier for learning algorithms)
    env = LogWrapper(env)  # Add stuff in the info dictionary for logging in the learning algo
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods
    env = NormalizeVecReward(env, gamma=0.99)  # normalize the reward in [-1, 1]
    env = NormalizeVecObservation(env)

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

    # @jit
    # def body(i, states_key):
    #     """Body of the fori_loop of play.
    #
    #     Args:
    #         i: number of the iteration.
    #         states_key: a tuple containing states and key.
    #     """
    #     states, key = states_key
    #
    #     key, *subkeys = random.split(key, env.num_drones + 1)
    #     actions = (
    #         jnp.array(
    #             [env.action_space(agent_id).sample(subkeys[agent_id]) for agent_id in range(env.num_drones)]
    #         )
    #         .flatten()
    #         .repeat(num_envs)
    #         .reshape((num_envs, env.num_drones, -1))
    #     )
    #
    #     print(actions)
    #
    #     key, *subkeys = random.split(key, num_envs + 1)
    #
    #     states = vmapped_step(actions, jnp.stack(subkeys), **env.state_to_dict(states))
    #
    #     # where you would learn or add to buffer
    #
    #     states = vmapped_auto_reset(**env.state_to_dict(states))
    #
    #     return (states, key)
    #
    # @jit
    # def play(key):
    #     """Execution of the environment with random actions."""
    #     key, *subkeys = random.split(key, num_envs + 1)
    #
    #     states = vmapped_reset(jnp.stack(subkeys))
    #
    #     states, key = jax.lax.fori_loop(0, 1000, body, (states, key))
    #
    #     return key, states
    #
    # key, states = play(key)  # compilation of the function
    #
    # durations = np.zeros(10)
    #
    # print("start")
    #
    # for i in range(10):
    #     start = time.time()
    #
    #     key, states = play(key)
    #
    #     jax.block_until_ready(states)
    #
    #     end = time.time() - start
    #
    #     durations[i] = end
    #
    # print("durations : ", durations)
