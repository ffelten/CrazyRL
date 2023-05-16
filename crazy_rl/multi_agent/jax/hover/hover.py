"""Hover environment for Crazyflies 2."""
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


class Hover(BaseParallelEnv):
    """A Parallel Environment where drone learn how to hover around a target point."""

    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        num_drones: int,
        init_flying_pos: jnp.ndarray,
        render_mode=None,
        size: int = 3,
        swarm=None,
    ):
        """Hover environment for Crazyflies 2.

        Args:
            num_drones: Number of drones
            init_flying_pos: Array of initial positions of the drones when they are flying
            render_mode: Render mode: "human", "real" or None
            size: Size of the map
            swarm: Swarm object, used for real tests. Ignored otherwise.
        """
        self.num_drones = num_drones

        self._init_flying_pos = init_flying_pos

        self.size = size

        self.norm = vmap(jnp.linalg.norm)  # function to compute the norm of each array in a matrix

        super().__init__(
            num_drones=num_drones,
            render_mode=render_mode,
            size=size,
            init_flying_pos=self._init_flying_pos,
            swarm=swarm,
        )

    @override
    def _observation_space(self, agent):
        return spaces.Box(
            low=np.array([-self.size, -self.size, 0, -self.size, -self.size, 0], dtype=np.float32),
            high=np.array([self.size, self.size, self.size, self.size, self.size, self.size], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

    @override
    def _action_space(self):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state, key):
        return jdc.replace(state, observations=vmap(jnp.append)(state.agents_locations, state.target_location)), key

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_action(self, state, actions):
        return jdc.replace(
            state,
            agents_locations=jnp.clip(
                state.agents_locations + actions * 0.2, jnp.array([-self.size, -self.size, 0]), self.size
            ),
        )

    @override
    @partial(jit, static_argnums=(0,))
    def _compute_reward(self, state):
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
            jnp.zeros(self.num_drones),  # terminations
            jnp.zeros(self.num_drones),  # truncations
            jnp.copy(self._init_flying_pos),  # target_location
        )


if __name__ == "__main__":
    parallel_env = Hover(
        num_drones=5,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
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
