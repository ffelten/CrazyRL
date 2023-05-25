"""The Base environment inheriting from pettingZoo Parallel environment class."""
import functools
from functools import partial
from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax_dataclasses as jdc
from gymnasium import spaces
from jax import jit
from pettingzoo.utils.env import ParallelEnv


@jdc.pytree_dataclass
class State:
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game
    observations: jnp.ndarray  # array containing the current observation of each agent
    rewards: jnp.ndarray  # array containing the current reward of each agent
    terminations: jnp.ndarray  # array of booleans which are True if the agents have crashed
    truncations: jnp.ndarray  # array of booleans which are True if the game reaches 100 timesteps


class BaseParallelEnv(ParallelEnv):
    """The Base environment inheriting from pettingZoo Parallel environment class.

    The main API methods of this class are:
    - step
    - reset

    They are defined in this main environment and the following compute methods must be implemented in child env:
        _action_space: Returns the Space object corresponding to valid actions
        _observation_space: Returns the Space object corresponding to valid observations
        _compute_obs: Computes the current observation of the environment.
        _compute_action: Computes the action passed to `.step()` into action matching the mode environment.
        _compute_reward: Computes the current reward value(s).
        _compute_terminated: Computes if the game must be stopped because the agents crashed.
        _compute_truncation: Computes if the game must be stopped because it is too long.
        _initialize_state: Initialize the State of the environment.
        auto_reset: Returns the State reinitialized if needed, else the actual State.
    """

    metadata = {
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: Optional[jnp.ndarray] = None,
        target_location: Optional[jnp.ndarray] = None,
        size: int = 3,
    ):
        """Initialization of a generic aviary environment.

        Args:
            num_drones: ids of the drones (ignored in simulation mode)
            init_flying_pos (array, optional): An array where each value is a (3)-shaped array containing the initial
                XYZ position of the drones.
            target_location (array, optional): An array containing a (3)-shaped array for the XYZ position of the target.
            size (int, optional): Size of the area sides
        """
        self.size = size  # The size of the square grid
        self._init_flying_pos = init_flying_pos
        self._target_location = target_location
        self.num_drones = num_drones

    def _observation_space(self, agent) -> spaces.Space:
        """Returns the observation space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _action_space(self) -> spaces.Space:
        """Returns the action space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_obs(self, state, key):
        """Computes the current observation of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_action(self, state, actions):
        """Computes the action passed to `.step()` into action matching the mode environment. Must be implemented in a subclass.

        Args:
            state : the state of the environment (contains agent_location used in this function)
            actions : ndarray. The input action for one drones
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
        """Creates a new states with initial values. Must be implemented in a subclass."""
        raise NotImplementedError

    def auto_reset(self, state):
        """Reset if needed (doesn't work)."""
        raise NotImplementedError

    # PettingZoo API
    @override
    @partial(jit, static_argnums=(0,))
    def reset(self, key, seed=None, return_info=False, options=None):
        return self._initialize_state()

    @override
    @partial(jit, static_argnums=(0,))
    def step(self, state, actions, key):
        state = self._compute_action(state, actions)

        state = jdc.replace(state, timestep=state.timestep + 1)

        state = self._compute_truncation(state)
        state = self._compute_terminated(state)
        state = self._compute_reward(state)
        (state,) = self._compute_obs(state, key)

        return state

    @override
    def state(self, state):
        states = jnp.array(
            [self._compute_obs(state.agent_location)[agent].astype(jnp.float32) for agent in range(self.num_drones)]
        )
        return jnp.concatenate(states, axis=None)

    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        return self._observation_space(agent)

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self):
        return self._action_space()
