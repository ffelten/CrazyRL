import jax.numpy as jnp
import jax.random as random

from crazy_rl.multi_agent.jax.surround.surround import Surround
from crazy_rl.utils.jax_wrappers import AutoReset, VecEnv


def test_vmap():
    """Test of the wrappers AutoReset and VecEnv to parallelize the Surround environment."""
    num_agents = 2
    env = Surround(
        num_drones=num_agents,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        target_location=jnp.array([[1.0, 0, 1.0]]),
    )

    num_envs = 2  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)

    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods

    key, *subkeys = random.split(key, num_envs + 1)

    obs, info, state = env.reset(jnp.stack(subkeys))

    # Different actions

    actions = jnp.array(
        [[[0, 1, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.2, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    key, *subkeys = random.split(key, num_envs + 1)

    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    assert (state.agents_locations == jnp.array([[[0, 0.2, 1], [0, 1, 1]], [[0.2, 0, 1], [0, 1, 1]]])).all()

    assert (
        obs
        == jnp.array(
            [
                [[0, 0.2, 1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1, 0, 0.2, 1]],
                [[0.2, 0, 1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1, 0.2, 0, 1]],
            ]
        )
    ).all()

    # Collision between two drones of the first State

    actions = jnp.array(
        [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.4, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    actions = jnp.array(
        [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.6, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    actions = jnp.array(
        [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.8, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    assert (term == jnp.array([[True, True], [False, False]])).all()

    assert (rewards == jnp.array([[-10, -10], [0, 0]])).all()

    assert (state.timestep == jnp.array([0, 4])).all()

    # Wait for the end of the game for the State 2

    actions = jnp.array(
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.8, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    for i in range(95):
        key, *subkeys = random.split(key, num_envs + 1)
        obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    assert (state.timestep == jnp.array([96, 0])).all()
    assert (trunc == jnp.array([[False, False], [True, True]])).all()

    # The reward isn't tested because it will probably change
