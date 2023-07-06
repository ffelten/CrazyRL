"""This is based on Gymnax's wrappers.py, but modified to work with our multi-agent environments."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State


class Wrapper:
    """Base class for wrappers."""

    def __init__(self, env: BaseParallelEnv):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class VecEnv(Wrapper):
    """Vectorized environment wrapper."""

    def __init__(self, env: BaseParallelEnv):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0,))
        self.step = jax.vmap(
            self._env.step,
            in_axes=(
                0,
                0,
                0,
            ),
        )
        self.state = jax.vmap(self._env.state, in_axes=(0,))


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: BaseParallelEnv):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, dict, LogEnvState]:
        obs, info, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, info, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: LogEnvState,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict, LogEnvState]:
        obs, rewards, terminateds, truncateds, info, env_state = self._env.step(state.env_state, action, key)
        done = jnp.logical_or(jnp.any(terminateds), jnp.any(truncateds))
        new_episode_return = state.episode_returns + rewards.sum()  # rewards are summed over agents "team reward"
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, rewards, terminateds, truncateds, info, state

    def state(self, state: LogEnvState) -> chex.Array:
        return self._env.state(state.env_state)


class AddIDToObs(Wrapper):
    """Add agent id to observation as one hot encoding."""

    def __init__(self, env, num_agents):
        super().__init__(env)
        self.num_agents = num_agents

    def _add_id(self, obs: jnp.ndarray) -> jnp.ndarray:
        # one hot encoding of agent id
        def _one_hot(id: int):
            return jnp.eye(self.num_agents)[id]

        return jnp.array([jnp.concatenate([o, _one_hot(id)]) for id, o in enumerate(obs)])

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, dict, State]:
        obs, info, state = self._env.reset(key)
        return self._add_id(obs), info, state

    def step(
        self, state: State, action: jnp.ndarray, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict, State]:
        obs, rewards, term, trunc, info, state = self._env.step(state, action, key)
        return self._add_id(obs), rewards, term, trunc, info, state

    def state(self, state: State) -> chex.Array:
        return self._env.state(state)


# TODO class AutoReset(Wrapper):
# see: https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py#L96C1-L123C65

# TODO class NormalizeObservation(Wrapper):
# see: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py#L193

# TODO class NormalizeReward(Wrapper):
# see: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py#L270
