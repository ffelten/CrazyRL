"""The Base environment inheriting from pettingZoo Parallel environment class."""
import functools

# import time
from functools import partial
from typing import Optional
from typing_extensions import override

import jax.numpy as jnp
import jax_dataclasses as jdc
import pygame
from cflib.crazyflie.swarm import Swarm
from gymnasium import spaces
from jax import jit, random
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_MODELVIEW_MATRIX,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_SMOOTH,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glColor4f,
    glColorMaterial,
    glEnable,
    glGetFloatv,
    glLight,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glMultMatrixf,
    glPopMatrix,
    glPushMatrix,
    glShadeModel,
)
from OpenGL.raw.GLU import gluLookAt, gluPerspective
from pettingzoo.utils.env import ParallelEnv
from pygame import DOUBLEBUF, OPENGL

from crazy_rl.utils.graphic import axes, field, point, target_point


# from crazy_rl.utils.utils import run_land, run_sequence, run_take_off


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
    - render
    - close

    They are defined in this main environment and the following compute methods must be implemented in child env:
        _action_space: Returns the Space object corresponding to valid actions
        _observation_space: Returns the Space object corresponding to valid observations
        _compute_obs: Computes the current observation of the environment.
        _compute_action: Computes the action passed to `.step()` into action matching the mode environment.
        _compute_reward: Computes the current reward value(s).
        _compute_terminated: Computes if the game must be stopped because the agents crashed.
        _compute_truncation: Computes if the game must be stopped because it is too long.
    """

    metadata = {
        "render_modes": ["human", "real"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: Optional[jnp.ndarray] = None,
        target_location: Optional[jnp.ndarray] = None,
        size: int = 3,
        render_mode: Optional[str] = None,
        swarm: Optional[Swarm] = None,
    ):
        """Initialization of a generic aviary environment.

        Args:
            num_drones: ids of the drones (ignored in simulation mode)
            init_flying_pos (array, optional): An array where each value is a (3)-shaped array containing the initial
                XYZ position of the drones.
            target_location (array, optional): An array containing a (3)-shaped array for the XYZ position of the target.
            size (int, optional): Size of the area sides
            render_mode (str, optional): The mode to display the rendering of the environment. Can be real, human or None.
                Real mode is used for real tests on the field, human mode is used to display the environment on a PyGame
                window and None mode is used to disable the rendering.
            swarm (Swarm, optional): The Swarm object use in real mode to control all drones
        """
        # State initialisation
        # self.state = state

        # Other initialisations
        self.size = size  # The size of the square grid
        self._init_flying_pos = init_flying_pos
        self._target_location = target_location
        self.num_drones = num_drones

        # Simulation and real drones initialisations
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._mode = "real" if self.render_mode == "real" else "simu"

        if self.render_mode == "human":
            self.window_size = 900  # The size of the PyGame window
            self.window = None
            self.clock = None
        """
        elif self.render_mode == "real":
            self.drone_ids = [i for i in range(num_drones)]
            assert swarm is not None, "Swarm object must be provided in real mode"
            self.swarm = swarm
            while not self.swarm:
                time.sleep(0.5)
                print("Waiting for connection...")
        """

    def _observation_space(self, agent) -> spaces.Space:
        """Returns the observation space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _action_space(self) -> spaces.Space:
        """Returns the action space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_obs(self, state):
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

    # PettingZoo API
    @override
    def reset(self, key, seed=None, return_info=False, options=None):
        """
        if self._mode == "real":
            # self.swarm.parallel_safe(reset_estimator)
            state = jdc.replace(state, agent_location=self._get_drones_state(self._mode, state.agent_location))
            print("reset", state.agent_location)

            command = dict()
            # dict target_position URI
            for agent in range(self.num_drones):
                uri = "radio://0/4/2M/E7E7E7E7" + str(agent).zfill(2)
                target = self._init_flying_pos[agent]
                agent = state.agent_location[agent]
                command[uri] = [[agent, target]]

            self.swarm.parallel_safe(run_take_off)
            print("Take off successful.")
            print(f"Setting the drone positions to the initial positions. {command}")
            self.swarm.parallel_safe(run_sequence, args_dict=command)

            state = jdc.replace(state, agent_location=self._get_drones_state(self._mode, state.agent_location))

        else:
        """

        key, subkey = random.split(key)

        state = self._initialize_state()

        state = self._compute_obs(state)

        if self.render_mode == "human" and self._mode == "simu":
            self._render_frame(state)

        return state, key

    @override
    @partial(jit, static_argnums=(0,))
    def step(self, state, actions, key):
        # state = jdc.replace(state, timestep=state.timestep + 1)

        # target_action = self._compute_action(actions, self._get_drones_state(self._mode, state.agent_location))

        """
        if self._mode == "real":
            command = dict()
            # dict target_position URI
            for agent in range(self.num_drones):
                uri = "radio://0/4/2M/E7E7E7E7" + str(agent).zfill(2)
                target = target_action[agent]
                current_location = state.agent_location[agent]
                command[uri] = [[current_location, target]]

            start = time.time()
            self.swarm.parallel_safe(run_sequence, args_dict=command)
            print("Time to execute the run_sequence", time.time() - start)

            state = jdc.replace(state, agent_location=self._get_drones_state(self._mode, state.agent_location))

        else:
        """

        key, subkey = random.split(key)

        state = self._compute_action(state, actions)

        """
        if self.render_mode == "human":
            self.render(state)
        """

        state = jdc.replace(state, timestep=state.timestep + 1)

        state = self._compute_truncation(state)
        state = self._compute_terminated(state)
        state = self._compute_reward(state)
        state = self._compute_obs(state)

        return state, key

    @override
    def render(self, state):
        if self.render_mode == "human" and self._mode == "simu":
            self._render_frame(state)

    def _render_frame(self, state):
        """Renders the current frame of the environment. Only works in human rendering mode."""

        def init_window():
            """Initializes the PyGame window."""
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Crazy RL")

            self.window = pygame.display.set_mode((self.window_size, self.window_size), DOUBLEBUF | OPENGL)

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glShadeModel(GL_SMOOTH)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            glLineWidth(1.5)

            glEnable(GL_LIGHT0)
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

            glMatrixMode(GL_PROJECTION)
            gluPerspective(45, (self.window_size / self.window_size), 0.1, 50.0)

            glMatrixMode(GL_MODELVIEW)
            gluLookAt(1, -10, 2, 0, 0, 0, 0, 0, 1)

            self.viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
            glLoadIdentity()

        if self.window is None and self.render_mode == "human":
            init_window()

        # if self.clock is None and self.render_mode == "human":
        self.clock = pygame.time.Clock()

        glLoadIdentity()

        # init the view matrix
        glPushMatrix()
        glLoadIdentity()

        # multiply the current matrix by the get the new view matrix and store the final view matrix
        glMultMatrixf(self.viewMatrix)
        self.viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)

        # apply view matrix
        glPopMatrix()
        glMultMatrixf(self.viewMatrix)

        glLight(GL_LIGHT0, GL_POSITION, (-1, -1, 5, 1))  # point light from the left, top, front

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for agent in state.agents_locations:
            glPushMatrix()
            point(jnp.copy(agent))

            glPopMatrix()

        glColor4f(0.5, 0.5, 0.5, 1)
        field(self.size)
        axes()

        for target in state.target_location:
            glPushMatrix()
            target_point(jnp.copy(target))
            glPopMatrix()

        pygame.event.pump()
        pygame.display.flip()

    @override
    def state(self, state):
        states = jnp.array(
            [self._compute_obs(state.agent_location)[agent].astype(jnp.float32) for agent in range(self.num_drones)]
        )
        return jnp.concatenate(states, axis=None)

    @override
    def close(self):
        if self._mode == "simu" and self.render_mode == "human":
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()
        """
        elif self._mode == "real":
            self.swarm.parallel_safe(run_land)
        """

    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        return self._observation_space(agent)

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self):
        return self._action_space()
