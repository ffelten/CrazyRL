"""The Base environment.

This class differs from its Numpy version because it cannot extend PettingZoo as jax is
based on functional programming while PZ relies heavily on OOP and state mutations.
"""
import functools
from functools import partial

import jax.numpy as jnp
import jax_dataclasses as jdc
from gymnasium import spaces
from jax import jit


@jdc.pytree_dataclass
class State:
    """State of the environment containing the modifiable variables.

    Jax is based on functional programming, this means we have to carry mutable variables along the way instead
    of hiding them as class members.
    In the same vein, the agents' states cannot be contained in a dictionary as it is not easily portable to GPU.
    Hence, we convert the dictionary of agents to arrays indexed from 0, based on the agent ids.
    """

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game.

    observations: jnp.ndarray  # array containing the current observation of each agent.
    rewards: jnp.ndarray  # array containing the current reward of each agent.
    terminations: jnp.ndarray  # array of booleans which are True if the agents have crashed.
    truncations: jnp.ndarray  # array of booleans which are True if the game reaches enough timesteps and ends.


class BaseParallelEnv:
    """The Base environment.

    The main API methods of this class is step.

    They are defined in this main environment and the following compute methods must be implemented in child env:
        _action_space: Returns the Space object corresponding to valid actions
        _observation_space: Returns the Space object corresponding to valid observations
        _compute_obs: Computes the current observation of the environment from a given state.
        _transition_state: Transitions the state based on the mechanics of the environment, for example makes the
                           target move.
        _sanitize_action: Makes the actions passed to step fit the environment, e.g. avoid making brutal moves.
        _compute_reward: Computes the current reward value(s) from a given state.
        _compute_terminated: Computes if the game must be stopped because the agents crashed from a given state.
        _compute_truncation: Computes if the game must be stopped because it is too long from a given state.
        _initialize_state: Initialize the State of the environment.
        reset: Resets the environment in initial state.
        auto_reset: Returns the State reinitialized if needed, else the actual State.
        state_to_dict: Translates the State into a dict.
        step_vmap: Used to vmap step, takes the values of the state and calls step with a new State object containing
                   the state values.
        state: Returns a global observation (concatenation of all the agent locations and the target locations).

    There are also the following functions:
        observation_space: Returns the observation space for one agent.
        action_space: Returns the action space for one agent.
    """

    def _observation_space(self, agent: int) -> spaces.Space:
        """Returns the observation space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _action_space(self, agent: int) -> spaces.Space:
        """Returns the action space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_obs(self, state: State) -> State:
        """Computes the current observation of the environment from a given state. Must be implemented in a subclass."""
        raise NotImplementedError

    def _transition_state(self, state: State, key: jnp.ndarray) -> State:
        """Transitions the state based on the mechanics of the environment, for example makes the target move. Must be implemented in a subclass."""
        raise NotImplementedError

    def _sanitize_action(self, state: State, actions: jnp.ndarray) -> State:
        """Makes the actions passed to step fit the environment, e.g. avoid making brutal moves. Must be implemented in a subclass.

        Args:
            state : the state of the environment (contains agent_location used in this function).
            actions : 2D array containing the x, y, z action for each drone.
        """
        raise NotImplementedError

    def _compute_reward(self, state: State) -> State:
        """Computes the current reward value(s) from a given state. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_terminated(self, state: State) -> State:
        """Computes if the game must be stopped because the agents crashed from a given state. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_truncation(self, state: State) -> State:
        """Computes if the game must be stopped because it is too long form a given state. Must be implemented in a subclass."""
        raise NotImplementedError

    def reset(self, key: jnp.ndarray) -> State:
        """Resets the environment in initial state. Must be implemented in a subclass."""
        raise NotImplementedError

    def auto_reset(self, state: State) -> State:
        """Returns the State reinitialized if needed, else the actual State. Must be implemented in a subclass."""
        raise NotImplementedError

    def state_to_dict(self, state: State) -> dict:
        """Translates the State into a dict. Must be implemented in a subclass."""
        raise NotImplementedError

    def step_vmap(self, action: jnp.ndarray, key: jnp.ndarray, **state_val) -> State:
        """Used to vmap step, takes the values of the state and calls step with a new State object containing the state values. Must be implemented in a subclass.

        Args:
            action: 2D array containing the x, y, z action for each drone.
            key : JAX PRNG key.
            **state_val: Different values contained in the State.
        """
        raise NotImplementedError

    def state(self, state: State) -> jnp.ndarray:
        """Returns a global observation (concatenation of all the agent locations and target locations). Must be implemented in a subclass."""
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def step(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        """Computes one step for the environment, in response to the actions of the drones."""
        state = self._sanitize_action(state, actions)

        state = jdc.replace(state, timestep=state.timestep + 1)

        state = self._compute_truncation(state)
        state = self._compute_terminated(state)
        state = self._transition_state(state, key)
        state = self._compute_reward(state)
        state = self._compute_obs(state)

        return state

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: int) -> spaces.Space:
        """Returns the observation space for one agent."""
        return self._observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int) -> spaces.Space:
        """Returns the action space for one agent."""
        return self._action_space(agent)
