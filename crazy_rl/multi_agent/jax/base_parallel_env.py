"""The Base environment inheriting from pettingZoo Parallel environment class."""
import functools
from functools import partial

import jax.numpy as jnp
import jax_dataclasses as jdc
from gymnasium import spaces
from jax import jit


@jdc.pytree_dataclass
class State:
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game.

    observations: jnp.ndarray  # array containing the current observation of each agent.
    rewards: jnp.ndarray  # array containing the current reward of each agent.
    terminations: jnp.ndarray  # array of booleans which are True if the agents have crashed.
    truncations: jnp.ndarray  # array of booleans which are True if the game reaches enough timesteps and ends.


class BaseParallelEnv:
    """The Base environment.

    The main API methods of this class are:
    - step
    - reset

    They are defined in this main environment and the following compute methods must be implemented in child env:
        _action_space: Returns the Space object corresponding to valid actions
        _observation_space: Returns the Space object corresponding to valid observations
        _compute_obs: Computes the current observation of the environment.
        _compute_mechanics: Computes the mechanics of the environment, for example the movements of the target.
        _compute_action: Computes the action passed to `.step()` into action matching the mode environment.
        _compute_reward: Computes the current reward value(s).
        _compute_terminated: Computes if the game must be stopped because the agents crashed.
        _compute_truncation: Computes if the game must be stopped because it is too long.
        _initialize_state: Initialize the State of the environment.
        auto_reset: Returns the State reinitialized if needed, else the actual State.
        state_to_dict: Translates the State into a dict.
        step_vmap: Calls step with a State and is called by vmap without State object.
        state: Returns a global observation (concatenation of all the agent locations and the target locations).

    There are also the following functions:
        observation_space: Returns the observation space for one agent.
        action_space: Returns the action space.
    """

    metadata = {
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def _observation_space(self, agent) -> spaces.Space:
        """Returns the observation space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _action_space(self, agent) -> spaces.Space:
        """Returns the action space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_obs(self, state):
        """Computes the current observation of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_mechanics(self, state, key):
        """Computes the mechanics of the environment, for example the movements of the target. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_action(self, state, actions):
        """Computes the action passed to `.step()` into action matching the mode environment. Must be implemented in a subclass.

        Args:
            state : the state of the environment (contains agent_location used in this function).
            actions : 2D array containing the x, y, z action for each drone.
        """
        raise NotImplementedError

    def _compute_reward(self, state):
        """Computes the current reward value(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_terminated(self, state):
        """Computes if the game must be stopped because the agents crashed. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_truncation(self, state):
        """Computes if the game must be stopped because it is too long. Must be implemented in a subclass."""
        raise NotImplementedError

    def _initialize_state(self):
        """Creates a new state with initial values. Must be implemented in a subclass."""
        raise NotImplementedError

    def auto_reset(self, state):
        """Returns the State reinitialized if needed, else the actual State. Must be implemented in a subclass."""
        raise NotImplementedError

    def state_to_dict(self, state):
        """Translates the State into a dict. Must be implemented in a subclass."""
        raise NotImplementedError

    def step_vmap(self, action, key, **state_val):
        """Calls step with a State and is called by vmap without State object. Must be implemented in a subclass."""
        raise NotImplementedError

    def state(self, state):
        """Returns a global observation (concatenation of all the agent locations and target locations). Must be implemented in a subclass."""
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        """Resets the environment in initial state."""
        state = self._initialize_state()
        state = self._compute_obs(state)
        return state

    @partial(jit, static_argnums=(0,))
    def step(self, state, actions, key):
        """Computes one step for the environment, in response to the actions of the drones."""
        state = self._compute_action(state, actions)

        state = jdc.replace(state, timestep=state.timestep + 1)

        state = self._compute_truncation(state)
        state = self._compute_terminated(state)
        state = self._compute_mechanics(state, key)
        state = self._compute_reward(state)
        state = self._compute_obs(state)

        return state

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Returns the observation space for one agent."""
        return self._observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Returns the action space."""
        return self._action_space(agent)
