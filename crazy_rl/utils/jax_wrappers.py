"""This is based on Gymnax's wrappers.py, but modified to work with our multi-agent environments."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from crazy_rl.multi_agent.jax.base_parallel_env import State


class Wrapper:
    """Base class for wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class VecEnv(Wrapper):
    """Vectorized environment wrapper."""

    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0,))
        self.step = jax.vmap(self._env.step)
        self.state = jax.vmap(self._env.state, in_axes=(0,))


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    total_timestep: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, total_timestep: int = 0) -> Tuple[chex.Array, dict, LogEnvState]:
        obs, info, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, total_timestep)
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
            total_timestep=state.total_timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["total_timestep"] = state.total_timestep
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


class AutoReset(Wrapper):
    """Automatically reset the environment when done.

    Based on Brax's wrapper; https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py#L96C1-L123C65"""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, dict, State]:
        return self._env.reset(key)

    def step(
        self, state: State, action: jnp.ndarray, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict, State]:
        obs, rewards, term, trunc, info, state = self._env.step(state, action, key)
        done = jnp.logical_or(jnp.any(term), jnp.any(trunc))

        def where_done(ifval, elseval):
            nonlocal done
            if done.shape:
                done = jnp.reshape(done, [ifval.shape[0]] + [1] * (len(elseval.shape) - 1))  # type: ignore
            return jnp.where(done, ifval, elseval)

        new_obs, new_info, new_state = self._env.reset(key, state.total_timestep)
        obs = where_done(new_obs, obs)
        state = jax.tree_util.tree_map(where_done, new_state, state)
        # TODO does not work with VecEnv... info["final_obs"] = where_done(new_obs, obs)
        return obs, rewards, term, trunc, info, state

    def state(self, state: State) -> chex.Array:
        return self._env.state(state)


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: State


class NormalizeVecReward(Wrapper):
    """Normalize the reward over a vectorized environment.

    Taken and adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py
    """

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, dict, NormalizeVecRewEnvState]:
        obs, info, state = self._env.reset(key)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, info, state

    def step(
        self, state: chex.Array, action: chex.Array, key: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict, NormalizeVecRewEnvState]:
        obs, reward, term, truncated, info, env_state = self._env.step(state.env_state, action, key)
        done = jnp.logical_or(jnp.any(term, axis=1), jnp.any(truncated, axis=1))
        return_val = state.return_val * self.gamma * (1 - done) + reward.sum(axis=1)  # team reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, reward / jnp.sqrt(state.var + 1e-8), term, truncated, info, state

    def state(self, state: NormalizeVecRewEnvState) -> chex.Array:
        return self._env.state(state.env_state)


class NormalizeObservation(Wrapper):
    """Normalize the observation.

    Taken and adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key):
        obs, info, state = self._env.reset(key)
        obs = obs / self._env.observation_space(0).high
        return obs, info, state

    def step(self, state, action, key):
        obs, reward, term, truncated, info, state = self._env.step(state, action, key)
        high = self._env.observation_space(0).high
        low = self._env.observation_space(0).low
        obs = -1 + (obs - low) * 2 / (high - low)  # min-max normalization
        return obs, reward, term, truncated, info, state

    def state(self, state: State) -> chex.Array:
        return self._env.state(state)


class ClipActions(Wrapper):
    """Clip actions to the action space."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, dict, State]:
        return self._env.reset(key)

    def step(
        self, state: State, action: jnp.ndarray, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict, State]:
        action = jnp.clip(action, self._env.action_space(0).low, self._env.action_space(0).high)
        return self._env.step(state, action, key)

    def state(self, state: State) -> chex.Array:
        return self._env.state(state)
