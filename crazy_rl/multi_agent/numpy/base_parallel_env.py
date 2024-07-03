"""The Base environment inheriting from pettingZoo Parallel environment class."""
import functools
import subprocess
import time
from copy import copy
from typing import Dict, Optional
from typing_extensions import override

import numpy as np
import numpy.typing as npt
import pygame
import zmq
from cflib.crazyflie.swarm import Swarm
from gymnasium import spaces
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
from crazy_rl.utils.utils import run_land, run_sequence, run_take_off


def _distance_to_target(agent_location: npt.NDArray[float], target_location: npt.NDArray[float]) -> float:
    return np.linalg.norm(agent_location - target_location)


CLOSENESS_THRESHOLD = 0.2


class BaseParallelEnv(ParallelEnv):
    """The Base environment inheriting from pettingZoo Parallel environment class.

    The main API methods of this class are:
    - step
    - reset
    - render
    - close
    - seed

    they are defined in this main environment and the following attributes can be set in child env through the compute
    method set:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """

    metadata = {
        "render_modes": ["human", "real"],
        "is_parallelizable": False,
        "render_fps": 10,
    }

    def __init__(
        self,
        agents_names: np.ndarray,
        drone_ids: np.ndarray,
        target_id: Optional[str] = None,
        init_flying_pos: Optional[Dict[str, np.ndarray]] = None,
        target_location: Optional[Dict[str, np.ndarray]] = None,
        size: int = 3,
        render_mode: Optional[str] = None,
        swarm: Optional[Swarm] = None,
    ):
        """Initialization of a generic aviary environment.

        Args:
            agents_names (list): list of agent names use as key for the dict
            drone_ids (list): ids of the drones (ignored in simulation mode)
            target_id (int, optional): ids of the targets (ignored in simulation mode). This is to control a real target with a real drone. Only supported in envs with one target.
            init_flying_pos (Dict, optional): A dictionary containing the name of the agent as key and where each value
                is a (3)-shaped array containing the initial XYZ position of the drones.
            target_location (Dict, optional): A dictionary containing a (3)-shaped array for the XYZ position of the target.
            size (int, optional): Size of the area sides
            render_mode (str, optional): The mode to display the rendering of the environment. Can be real, human or None.
                Real mode is used for real tests on the field, human mode is used to display the environment on a PyGame
                window and None mode is used to disable the rendering.
            swarm (Swarm, optional): The Swarm object use in real mode to control all drones
        """
        self.size = size  # The size of the square grid
        self._agent_location = init_flying_pos.copy()
        self._previous_location = init_flying_pos.copy()  # for potential based reward
        self._init_flying_pos = init_flying_pos
        self._init_target_location = target_location
        self._target_location = target_location
        self._previous_target = target_location.copy()
        self.possible_agents = agents_names.tolist()
        self.timestep = 0
        self.agents = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._mode = "real" if self.render_mode == "real" else "simu"

        if self.render_mode == "human":
            self.window_size = 900  # The size of the PyGame window
            self.window = None
            self.clock = None
        elif self.render_mode == "real":
            self.drone_ids = drone_ids
            self.target_id = target_id
            assert swarm is not None, "Swarm object must be provided in real mode"
            self.swarm = swarm
            while not self.swarm:
                time.sleep(0.5)
                print("Waiting for connection...")
        elif self.render_mode == "unity":
            self.drone_ids = drone_ids
            self.server = False
            self.socket = None
            self.i = 0
            self.is_setup = False

    def _observation_space(self, agent) -> spaces.Space:
        """Returns the observation space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _action_space(self, agent) -> spaces.Space:
        """Returns the action space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_obs(self):
        """Returns the current observation of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _transition_state(self, action):
        """Computes the action passed to `.step()` into action matching the mode environment. Must be implemented in a subclass.

        Args:
            action : ndarray | dict[..]. The input action for one drones
        """
        raise NotImplementedError

    def _compute_reward(self):
        """Computes the current reward value(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_terminated(self):
        """Computes the current done value(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_truncation(self):
        """Computes the current done value(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_info(self):
        """Computes the current info dict(s). Must be implemented in a subclass."""
        raise NotImplementedError

    # PettingZoo API
    @override
    def reset(self, seed=None, return_info=False, options=None):
        self.timestep = 0
        self.agents = copy(self.possible_agents)
        self._target_location = self._init_target_location.copy()
        self._previous_target = self._init_target_location.copy()

        if self._mode == "simu":
            self._agent_location = self._init_flying_pos.copy()
            self._previous_location = self._init_flying_pos.copy()
        elif self._mode == "real":
            # self.swarm.parallel_safe(reset_estimator)
            target_loc, self._agent_location = self._get_drones_state()
            self._previous_location = self._agent_location.copy()
            print("reset", self._agent_location)

            command = dict()
            # dict target_position URI
            for id in self.drone_ids:
                uri = "radio://0/4/2M/E7E7E7E7" + str(id).zfill(2)
                next_loc = self._init_flying_pos["agent_" + str(id)]
                current_loc = self._agent_location["agent_" + str(id)]
                command[uri] = [[current_loc, next_loc]]

            # Move target drone into position
            if self.target_id is not None:
                uri = "radio://0/4/2M/E7E7E7E7" + str(self.target_id).zfill(2)
                current = target_loc
                target = list(self._init_target_location.values())[0]
                command[uri] = [[current, target]]

            self.swarm.parallel_safe(run_take_off)
            print("Take off successful.")
            print(f"Setting the drone positions to the initial positions. {command}")
            self.swarm.parallel_safe(run_sequence, args_dict=command)

            target_loc, self._agent_location = self._get_drones_state()
            if self.target_id is not None:
                self._target_location = {"unique": target_loc}

        observation = self._compute_obs()
        infos = self._compute_info()

        if self.render_mode == "human" and self._mode == "simu":
            self._render_frame()

        return observation, infos

    @override
    def step(self, actions):
        self.timestep += 1

        if self._mode == "simu":
            self.render()
            new_locations = self._transition_state(actions)
            self._previous_location = self._agent_location
            self._agent_location = new_locations

        elif self._mode == "real":
            new_locations = self._transition_state(actions)
            command = dict()
            # dict target_position URI
            for id in self.drone_ids:
                uri = "radio://0/4/2M/E7E7E7E7" + str(id).zfill(2)
                target = new_locations["agent_" + str(id)]
                current_location = self._agent_location["agent_" + str(id)]
                command[uri] = [[current_location, target]]

            if self.target_id is not None:
                uri = "radio://0/4/2M/E7E7E7E7" + str(self.target_id).zfill(2)
                current = list(self._previous_target.values())[0]
                target = list(self._target_location.values())[0]
                command[uri] = [[current, target]]

            start = time.time()
            self.swarm.parallel_safe(run_sequence, args_dict=command)
            print("Time to execute the run_sequence", time.time() - start)

            # (!) Updates of location are not relying on cflib because it is too slow in practice
            # So yes, we assume the drones go where we tell them to go
            self._previous_location = self._agent_location
            self._agent_location = new_locations

        terminations = self._compute_terminated()
        truncations = self._compute_truncation()
        rewards = self._compute_reward()
        observations = self._compute_obs()
        infos = self._compute_info()

        return observations, rewards, terminations, truncations, infos

    @override
    def render(self):
        if self.render_mode == "human" and self._mode == "simu":
            self._render_frame()
        if self.render_mode == "unity":
            self.send_to_unity()

    def _render_frame(self):
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
            gluPerspective(75, (self.window_size / self.window_size), 0.1, 50.0)

            glMatrixMode(GL_MODELVIEW)
            gluLookAt(3, -11, 3, 0, 0, 0, 0, 0, 1)

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

        for agent in self._agent_location.values():
            glPushMatrix()
            point(np.array([agent[0], agent[1], agent[2]]))

            glPopMatrix()

        glColor4f(0.5, 0.5, 0.5, 1)
        field(self.size)
        axes()

        for target in self._target_location.values():
            glPushMatrix()
            target_point(np.array([target[0], target[1], target[2]]))
            glPopMatrix()

        pygame.event.pump()
        pygame.display.flip()

    @override
    def state(self):
        states = tuple(self._compute_obs()[agent].astype(np.float32) for agent in self.possible_agents)
        return np.concatenate(states, axis=None)

    @override
    def close(self):
        if self._mode == "simu" and self.render_mode == "human":
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()
        elif self._mode == "real":
            self.swarm.parallel_safe(run_land)

    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        return self._observation_space(agent)

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self, agent):
        return self._action_space(agent)

    def _get_drones_state(self):
        """Return the state of all drones (xyz position) inside a dict with the same keys of agent_location and target_location."""
        if self._mode == "simu":
            return list(self._target_location.values()), self._agent_location
        elif self._mode == "real":
            agent_locs = dict()
            target_loc = None
            pos = self.swarm.get_estimated_positions()
            for uri in pos:
                if self.target_id is not None and uri[-1] == self.target_id:
                    target_loc = np.array(pos[uri])
                else:
                    agent_locs["agent_" + uri[-1]] = np.array(pos[uri])

            return target_loc, agent_locs

    def send_to_unity(self):
        """Starts the unity executable and sends it the data.

        By default, the application runs under Linux. To run under Window, uncomment the subprocess.Popen(... .exe) line and comment out the subprocess.Popen(... .x86_64) line.
        """

        def init_serv():
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind("tcp://*:5555")
            self.server = True
            self.is_setup = False

        def send_env():
            self.is_setup = True
            data = {
                "nbDrones": len(self.drone_ids),
                "size": self.size * 5,
                "type": "init",
            }
            self.socket.recv()
            # Send reply back to client (data)
            self.socket.send_json(data)

        def send_pos(point, type, id):
            """In Unity, the Y axis is up, whereas in the code, the Z axis is up.

            To get the right coordinates in Unity, you need to invert the Y and Z coordinates,
            so posY = z and posZ = y.
            """
            data = {
                "id": int(id),
                "posX": float(point[0] * 5),
                "posY": float(point[2] * 5),
                "posZ": float(point[1] * 5),
                "type": type,
            }
            self.socket.recv()
            # Send reply back to client (data)
            self.socket.send_json(data)

        if self.server is False and self.render_mode == "unity":
            init_serv()
            """run with Linux"""
            subprocess.Popen("./crazy_rl/multi_agent/numpy/bin/unity/Linux/CrazyRlUnity.x86_64")
            """run with Window"""
            # subprocess.Popen("./crazy_rl/multi_agent/numpy/bin/unity/Window/CrazyRl_Unity.exe")

        if self.server:
            if not self.is_setup:
                send_env()
            else:
                self.i = 0
                for agent in self._agent_location.values():
                    send_pos(np.array([agent[0], agent[1], agent[2]]), "Drone", self.drone_ids[self.i])
                    self.i += 1

                for target in self._target_location.values():
                    send_pos(np.array([target[0], target[1], target[2]]), "Target", len(self.drone_ids))
