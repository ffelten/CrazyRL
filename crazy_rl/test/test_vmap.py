import jax.numpy as jnp
import jax.random as random
from jax import vmap

from crazy_rl.multi_agent.jax.surround.surround import Surround


def test_vmap():
    """Test for the parallelization of Surround environment with vmap."""
    parallel_env = Surround(
        num_drones=2,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        target_location=jnp.array([[1.0, 0, 1.0]]),
    )

    n = 2  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)

    key, *subkeys = random.split(key, n + 1)

    states = vmap(parallel_env.reset)(jnp.stack(subkeys))

    # Different actions

    actions = jnp.array(
        [[[0, 1, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.2, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    key, *subkeys = random.split(key, n + 1)

    states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))

    assert (states.agents_locations == jnp.array([[[0, 0.2, 1], [0, 1, 1]], [[0.2, 0, 1], [0, 1, 1]]])).all()

    states = vmap(parallel_env.auto_reset)(**parallel_env.state_to_dict(states))

    assert (
        states.observations
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

    key, *subkeys = random.split(key, n + 1)
    states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))
    states = vmap(parallel_env.auto_reset)(**parallel_env.state_to_dict(states))

    actions = jnp.array(
        [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.6, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    key, *subkeys = random.split(key, n + 1)
    states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))
    states = vmap(parallel_env.auto_reset)(**parallel_env.state_to_dict(states))

    actions = jnp.array(
        [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.8, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    key, *subkeys = random.split(key, n + 1)
    states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))

    assert (states.terminations == jnp.array([[True, True], [False, False]])).all()

    assert (states.rewards == jnp.array([[-10, -10], [0, 0]])).all()

    states = vmap(parallel_env.auto_reset)(**parallel_env.state_to_dict(states))

    assert (states.timestep == jnp.array([0, 4])).all()

    # Wait for the end of the game for the State 2

    actions = jnp.array(
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # state 1 : [[0, 0.8, 1], [0, 1, 1]]
    )  # state 2 : [[0.2, 0, 1], [0, 1, 1]]

    for i in range(95):
        key, *subkeys = random.split(key, n + 1)
        states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))
        states = vmap(parallel_env.auto_reset)(**parallel_env.state_to_dict(states))

    states = vmap(parallel_env.step_vmap)(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))

    assert (states.timestep == jnp.array([96, 100])).all()
    assert (states.truncations == jnp.array([[False, False], [True, True]])).all()

    # The reward isn't tested because it will probably change
