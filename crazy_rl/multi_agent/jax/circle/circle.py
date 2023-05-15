"""Circle environment for Crazyflie 2. Each agent is supposed to learn to perform a circle around a target point."""
import time
from functools import partial
from typing_extensions import override

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from gymnasium import spaces
from jax import jit, random, vmap

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv


@override
@jdc.pytree_dataclass
class State:
    """State of the environment containing the modifiable variables."""

    agents_locations: jnp.ndarray  # a 2D array containing x,y,z coordinates of each agent, indexed from 0.
    timestep: int  # represents the number of steps already done in the game

    observations: jnp.ndarray  # array containing the current observation of each agent
    rewards: jnp.ndarray  # array containing the current reward of each agent
    terminations: jnp.ndarray  # array of booleans which are True if the agents have crashed
    truncations: jnp.ndarray  # array of booleans which are True if the game reaches 100 timesteps

    target_location: jnp.ndarray  # 2D array containing x,y,z coordinates of the target of each agent


class Circle(BaseParallelEnv):
    """A Parallel Environment where drone learn how to perform a circle."""

    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        num_intermediate_points: int = 10,
        render_mode=None,
        size: int = 3,
        swarm=None,
    ):
        """Circle environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            num_intermediate_points: Number of intermediate points in the target circle
            render_mode: Render mode: "human", "real" or None
            size: Size of the map
            swarm: Swarm object, used for real tests. Ignored otherwise.
        """
        self.num_drones = num_drones

        self.size = size

        self._init_flying_pos = init_flying_pos

        self.norm = vmap(jnp.linalg.norm)  # function to compute the norm of each array in a matrix

        # Specific to circle

        circle_radius = 0.5  # [m]

        self.num_intermediate_points = num_intermediate_points

        # Ref is a list of 2d arrays for each agent
        # each 2d array contains the reference points (xyz) for the agent at each timestep
        self.ref = jnp.zeros((num_intermediate_points, self.num_drones, 3))

        ts = 2 * np.pi * np.arange(num_intermediate_points) / num_intermediate_points

        for agent in range(self.num_drones):
            self.ref = self.ref.at[:, agent, 0].set(
                circle_radius * (1 - np.cos(ts)) + (init_flying_pos[agent][0] - circle_radius)
            )
            self.ref = self.ref.at[:, agent, 1].set(init_flying_pos[agent][1])
            self.ref = self.ref.at[:, agent, 2].set(circle_radius * np.sin(ts) + (init_flying_pos[agent][2]))

        super().__init__(
            render_mode=render_mode,
            size=size,
            init_flying_pos=self._init_flying_pos,
            num_drones=self.num_drones,
            swarm=swarm,
        )

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=jnp.array([-self.size, -self.size, 0, -self.size, -self.size, 0], dtype=jnp.float32),
            high=jnp.array([self.size, self.size, self.size, self.size, self.size, self.size], dtype=jnp.float32),
            shape=(6,),
            dtype=jnp.float32,
        )

    @override
    def _action_space(self):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state):
        target_location = self.ref[state.timestep % self.num_intermediate_points]  # redo the circle if the end is reached

        return jdc.replace(
            state, observations=vmap(jnp.append)(state.agents_locations, target_location), target_location=target_location
        )

    @override
    def _compute_action(self, state, actions):
        # Actions are clipped to stay in the map and scaled to do max 20cm in one step
        return jdc.replace(
            state,
            agents_locations=jnp.clip(
                state.agents_locations + actions * 0.2, jnp.array([-self.size, -self.size, 0]), self.size
            ),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state):
        # Reward is based on the euclidean distance to the target point

        return jdc.replace(state, rewards=-1 * self.norm(state.target_location - state.agents_locations))

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state):
        # the drones never crash. Terminations initialized to jnp.zeros() and then never changes
        return state

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state):
        return jdc.replace(state, truncations=(state.timestep == 100) * jnp.ones(self.num_drones))

    @override
    @partial(jit, static_argnums=(0,))
    def _initialize_state(self):
        return State(
            self._init_flying_pos,
            0,
            jnp.array([]),
            jnp.array([]),
            jnp.zeros(self.num_drones),
            jnp.zeros(self.num_drones),
            jnp.array([]),
        )


if __name__ == "__main__":
    parallel_env = Circle(
        num_drones=5,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        num_intermediate_points=100,
    )

    # to verify the proportion of crash and avoid some mistakes
    nb_crash = 0
    nb_end = 0

    global_step = 0
    start_time = time.time()

    seed = 5

    key = random.PRNGKey(seed)

    state, key = parallel_env.reset(key)

    parallel_env.render(state)

    for i in range(100):
        while not jnp.any(state.truncations) and not jnp.any(state.terminations):
            actions = jnp.array([parallel_env.action_space().sample() for _ in range(parallel_env.num_drones)])

            state, key = parallel_env.step(state, actions, key)

            parallel_env.render(state)
            # print("obs", observations, "reward", rewards)

            if global_step % 2000 == 0:
                print("SPS:", int(global_step / (time.time() - start_time)))

            global_step += 1

            # time.sleep(0.02)

        nb_crash += jnp.any(state.terminations)
        nb_end += jnp.any(state.truncations)

        state, key = parallel_env.reset(key)

    print("nb_crash", nb_crash)
    print("nb_end", nb_end)
    print("total", nb_end + nb_crash)
