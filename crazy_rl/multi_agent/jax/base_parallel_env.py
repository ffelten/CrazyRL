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
        action_space: Returns the Space object corresponding to valid actions for one agent.
        observation_space: Returns the Space object corresponding to valid observations for one agent.
        _compute_obs: Computes the current observation of the environment from a given state.
        _transition_state: Transitions the state based on the mechanics of the environment, for example makes the
                           target move and sanitize the actions.
        _compute_reward: Computes the current reward value(s) from a given state.
        _compute_terminated: Computes if the game must be stopped because the agents crashed from a given state.
        _compute_truncation: Computes if the game must be stopped because it is too long from a given state.
        reset: Resets the environment in initial state.
        auto_reset: Returns the State reinitialized if needed, else the actual State.
        state_to_dict: Translates the State into a dict.
        step_vmap: Used to vmap step, takes the values of the state and calls step with a new State object containing
                   the state values.
        state: Returns a global observation (concatenation of all the agent locations and the target locations).

    There are also the following functions:
        _sanitize_action: Makes the actions passed to step fit the environment, e.g. avoid making brutal moves.
    """

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: int) -> spaces.Space:
        """Returns the observation space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: int) -> spaces.Space:
        """Returns the action space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_obs(self, state: State) -> State:
        """Computes the current observation of the environment from a given state. Must be implemented in a subclass."""
        raise NotImplementedError

    def _transition_state(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        """Transitions the state based on the mechanics of the environment, for example makes the target move and sanitize the actions. Must be implemented in a subclass."""
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _sanitize_action(self, state: State, actions: jnp.ndarray) -> jnp.ndarray:
        """Makes the actions passed to step fit the environment, e.g. avoid making brutal moves.

        Args:
            state : the state of the environment (contains agent_location used in this function).
            actions : 2D array containing the x, y, z action for each drone.
        """
        # Actions are clipped to stay in the map and scaled to do max 20cm in one step
        return jnp.clip(state.agents_locations + actions * 0.2, jnp.array([-self.size, -self.size, 0]), self.size)

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

    def auto_reset(self, **state) -> State:
        """Returns the State reinitialized if needed, else the actual State. Must be implemented in a subclass.

        The values contained by State are passed in argument and used like a dictionary
        because auto_reset is meant to be used by vmap and vmap doesn't accept objects.

        This function handles states like a dictionary, see also the pydoc of step_vmap.
        """
        raise NotImplementedError

    def step_vmap(self, action: jnp.ndarray, key: jnp.ndarray, **state_val) -> State:
        """Used to vmap step.

         Takes the values of the state and calls step with a new State object containing
         the state values. Must be implemented in a subclass.

         JAX's vmap cannot operate on array-of-structs, but can operate on struct-of-arrays,
         so the states actually contain array of arrays after vmap. Our solution to this is
         to convert the struct into a dictionary of array of arrays and plug it into the vmapped
         function as kwargs. This way, each value of the kwargs (the state members) will be
         processed as a regular array.

        Args:
            action: 2D array containing the x, y, z action for each drone.
            key : JAX PRNG key.
            **state_val: Different values contained in the State.
        """
        raise NotImplementedError

    def state_to_dict(self, state: State) -> dict:
        """Translates the State into a dict. Must be implemented in a subclass."""
        raise NotImplementedError

    def state(self, state: State) -> jnp.ndarray:
        """Returns a global observation (concatenation of all the agent locations and target locations). Must be implemented in a subclass."""
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def step(self, state: State, actions: jnp.ndarray, key: jnp.ndarray) -> State:
        """Computes one step for the environment, in response to the actions of the drones."""
        state = jdc.replace(state, timestep=state.timestep + 1)

        state = self._transition_state(state, actions, key)

        state = self._compute_truncation(state)
        state = self._compute_terminated(state)
        state = self._compute_reward(state)
        state = self._compute_obs(state)

        return state
