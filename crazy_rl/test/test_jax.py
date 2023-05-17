import jax.numpy as jnp
import jax.random as random

from crazy_rl.multi_agent.jax.surround.surround import Surround


def test_jax_surround():
    """Test for the Surround environment in jax version."""
    parallel_env = Surround(
        num_drones=2,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
    )

    seed = 5

    key = random.PRNGKey(seed)

    # 1st round : the two drones crash

    state, key = parallel_env.reset(key)

    actions = jnp.array([[0, 1, 0], [0, -1, 0]])

    state, key = parallel_env.step(state, actions, key)

    assert (state.agents_locations == jnp.array([[0, 0.2, 1], [0, 0.8, 1]])).all()

    actions = jnp.array([[0, 1, 0], [0, -1, 0]])

    state, key = parallel_env.step(state, actions, key)

    assert not state.terminations.any()

    actions = jnp.array([[0, 1, 0], [0, -1, 0]])

    state, key = parallel_env.step(state, actions, key)

    # the drones crash
    assert state.terminations.all()

    assert (state.rewards == jnp.array([-10, -10])).all()

    # 2nd round : one drone crashes with the target

    state, key = parallel_env.reset(key)

    actions = jnp.array([[0, 0, 0], [1, 0, 1]])  # position = [0.2, 1, 1.2]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, 0], [1, 0, 1]])  # position = [0.4, 1, 1.4]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, 0], [1, 0, 1]])  # position = [0.6, 1, 1.6]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, 0], [1, 0, 1]])  # position = [0.8, 1, 1.8]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, 0], [1, 0, 1]])  # position = [1, 1, 2]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, 0], [0, 0, 1]])  # position = [1, 1, 2.2]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, 0], [0, 0, 1]])  # position = [1, 1, 2.4]
    state, key = parallel_env.step(state, actions, key)

    # the drone crashes in the target
    assert state.terminations.all()

    # 3rd round : one drone crashes with the ground

    state, key = parallel_env.reset(key)

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.8]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.6]
    state, key = parallel_env.step(state, actions, key)

    assert (state.observations == jnp.array([[0, 0, 0.6, 1, 1, 2.5, 0, 1, 1], [0, 1, 1, 1, 1, 2.5, 0, 0, 0.6]])).all()

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.4]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.2] (0.20003)
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, -0.01], [0, 0, 0]])  # position = [0, 0, 0.2] (just below)
    state, key = parallel_env.step(state, actions, key)

    # the drone crashes on the ground
    assert state.terminations.all()

    # 4th round : the drones never crash and one tries to leave the map

    state, key = parallel_env.reset(key)

    for i in range(10):
        actions = jnp.array([[0, 0, 0], [0, 1, 0]])
        state, key = parallel_env.step(state, actions, key)

    # the drone stays in the map
    assert state.agents_locations[1, 1] <= 3

    actions = jnp.array([[0, 0, 0], [0, 0, 0]])

    for i in range(90):
        state, key = parallel_env.step(state, actions, key)

    # the game ends after 100 timesteps
    assert state.truncations.all()

    assert (
        state.rewards
        == (6 - jnp.array([jnp.linalg.norm(jnp.array([1, 1, 1.5])), jnp.linalg.norm(jnp.array([1, 2, 1.5]))])) * 0.95
        + jnp.linalg.norm(jnp.array([0, 3, 0])) * 0.05
    ).all()
