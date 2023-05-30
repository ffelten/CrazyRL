import jax.numpy as jnp
import jax.random as random

from crazy_rl.multi_agent.jax.catch.catch import Catch
from crazy_rl.multi_agent.jax.circle.circle import Circle
from crazy_rl.multi_agent.jax.escort.escort import Escort
from crazy_rl.multi_agent.jax.hover.hover import Hover
from crazy_rl.multi_agent.jax.surround.surround import Surround


def two_drones_crash(parallel_env, key):
    """Test where two drones starting on (0, 0, 1), (0, 1, 1) crash together."""
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

    return state, key


def crash_ground(parallel_env, key):
    """Test where the first drone, starting on (0, 0, 1), crashes on the ground."""
    state, key = parallel_env.reset(key)

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.8]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.6]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.4]
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, -1], [0, 0, 0]])  # position = [0, 0, 0.2] (0.20003)
    state, key = parallel_env.step(state, actions, key)

    actions = jnp.array([[0, 0, -0.01], [0, 0, 0]])  # position = [0, 0, 0.2] (just below)
    state, key = parallel_env.step(state, actions, key)

    # the drone crashes on the ground
    assert state.terminations.all()

    return state, key


def leave_the_map(parallel_env, key):
    """Test where the second drone, starting on (0, 1, 1) tries to leave the map, and they stay alive until the
    end of the round."""
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

    return state, key


def test_surround():
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

    state, key = two_drones_crash(parallel_env, key)

    # 2nd round : one drone crashes with the target

    state, key = parallel_env.reset(key)

    actions = jnp.array([[0, 0, 0], [1, 0, 1]])  # position = [0.2, 1, 1.2]
    state, key = parallel_env.step(state, actions, key)

    assert (state.observations == jnp.array([[0, 0, 1, 1, 1, 2.5, 0.2, 1, 1.2], [0.2, 1, 1.2, 1, 1, 2.5, 0, 0, 1]])).all()

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

    state, key = crash_ground(parallel_env, key)

    # 4th round : the drones never crash and one tries to leave the map

    state, key = leave_the_map(parallel_env, key)

    assert (
        state.rewards
        == (6 - jnp.array([jnp.linalg.norm(jnp.array([1, 1, 1.5])), jnp.linalg.norm(jnp.array([1, 2, 1.5]))])) * 0.95
        + jnp.linalg.norm(jnp.array([0, 3, 0])) * 0.05
    ).all()


def test_hover():
    """Test for the Hover environment in jax version."""
    parallel_env = Hover(
        num_drones=2,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
    )

    seed = 5

    key = random.PRNGKey(seed)

    state, key = leave_the_map(parallel_env, key)

    # observation : agent's location and target's location
    assert (state.observations == jnp.array([[0, 0, 1, 0, 0, 1], [0, 3, 1, 0, 1, 1]])).all()

    assert (state.rewards == jnp.array([0, -2])).all()


def test_circle():
    """Test for the Circle environment in jax version."""
    parallel_env = Circle(
        num_drones=2,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        num_intermediate_points=100,
    )

    seed = 5

    key = random.PRNGKey(seed)

    state, key = leave_the_map(parallel_env, key)
    # observation : agent's location and target's location
    assert (state.observations == jnp.array([[0, 0, 1, -0.5, 0, 1], [0, 3, 1, -0.5, 1, 1]])).all()

    assert (state.rewards > jnp.array([-0.5, -jnp.linalg.norm(jnp.array([0.5, 2, 0]))]) - 0.05).all()
    assert (state.rewards < jnp.array([-0.5, -jnp.linalg.norm(jnp.array([0.5, 2, 0]))]) + 0.01).all()

    state, key = parallel_env.reset(key)

    actions = jnp.zeros((2, 3))

    state, key = parallel_env.step(state, actions, key)

    ts = 2 * jnp.pi / 100

    assert (
        state.target_location
        == jnp.array(
            [
                [0.5 * (1 - jnp.cos(ts)) - 0.5, 0.5 * jnp.sin(ts), 1],
                [0.5 * (1 - jnp.cos(ts)) - 0.5, 0.5 * jnp.sin(ts) + 1, 1],
            ]
        )
    ).all()


def test_escort():
    """Test for the Escort environment in jax version."""
    parallel_env = Escort(
        num_drones=2,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        init_target_location=jnp.array([1, 1, 2.5]),
        final_target_location=jnp.array([-2, -2, 3]),
        num_intermediate_points=150,
    )

    seed = 5

    key = random.PRNGKey(seed)

    # 1st round : the two drones crash

    state, key = two_drones_crash(parallel_env, key)

    # 2nd round : one drone crashes with the ground

    state, key = crash_ground(parallel_env, key)

    # 3rd round : the drones never crash and one tries to leave the map

    state, key = leave_the_map(parallel_env, key)

    # observation : agent's location, target's location and other agent's location
    assert (
        state.observations
        == jnp.array(
            [[0, 0, 1, -0.9736842, -0.9736842, 2.8289473, 0, 3, 1], [0, 3, 1, -0.9736842, -0.9736842, 2.8289473, 0, 0, 1]]
        )
    ).all()

    # not exactly the same due to approximations
    assert (
        state.rewards
        > (
            6
            - jnp.array(
                [
                    jnp.linalg.norm(jnp.array([0, 0, 1]) - jnp.array([-0.9736842, -0.9736842, 2.8289473])),
                    jnp.linalg.norm(jnp.array([0, 3, 1]) - jnp.array([-0.9736842, -0.9736842, 2.8289473])),
                ]
            )
        )
        * 0.95
        + jnp.linalg.norm(jnp.array([0, 3, 0])) * 0.05
        - 0.05
    ).all()

    assert (
        state.rewards
        < (
            6
            - jnp.array(
                [
                    jnp.linalg.norm(jnp.array([0, 0, 1]) - jnp.array([-0.9736842, -0.9736842, 2.8289473])),
                    jnp.linalg.norm(jnp.array([0, 3, 1]) - jnp.array([-0.9736842, -0.9736842, 2.8289473])),
                ]
            )
        )
        * 0.95
        + jnp.linalg.norm(jnp.array([0, 3, 0])) * 0.05
        + 0.05
    ).all()

    state, key = parallel_env.reset(key)

    actions = jnp.zeros((2, 3))

    state, key = parallel_env.step(state, actions, key)

    assert (state.target_location == jnp.array([1, 1, 2.5]) + jnp.array([-3, -3, 0.5]) / 152).all()


def test_catch():
    """Test for the Catch environment in jax version."""
    parallel_env = Catch(
        num_drones=2,
        render_mode=None,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        init_target_location=jnp.array([1, 1, 2.5]),
        target_speed=0.1,
    )

    seed = 5

    key = random.PRNGKey(seed)

    # 1st round : the two drones crash

    state, key = two_drones_crash(parallel_env, key)

    # 2nd round : one drone crashes with the ground

    state, key = crash_ground(parallel_env, key)

    # 3rd round : the drones never crash and one tries to leave the map

    state, key = leave_the_map(parallel_env, key)

    # observation : agent's location, target's location and other agent's location
    assert (
        state.observations
        == jnp.array(
            [
                [0, 0, 1, state.target_location[0, 0], state.target_location[0, 1], state.target_location[0, 2], 0, 3, 1],
                [0, 3, 1, state.target_location[0, 0], state.target_location[0, 1], state.target_location[0, 2], 0, 0, 1],
            ]
        )
    ).all()

    # not exactly the same due to approximations
    assert (
        state.rewards
        > (
            6
            - jnp.array(
                [
                    jnp.linalg.norm(jnp.array([0, 0, 1]) - state.target_location),
                    jnp.linalg.norm(jnp.array([0, 3, 1]) - state.target_location),
                ]
            )
        )
        * 0.95
        + jnp.linalg.norm(jnp.array([0, 3, 0])) * 0.05
        - 0.1
    ).all()

    assert (
        state.rewards
        < (
            6
            - jnp.array(
                [
                    jnp.linalg.norm(jnp.array([0, 0, 1]) - state.target_location),
                    jnp.linalg.norm(jnp.array([0, 3, 1]) - state.target_location),
                ]
            )
        )
        * 0.95
        + jnp.linalg.norm(jnp.array([0, 3, 0])) * 0.05
        + 0.1
    ).all()

    # Tests the target

    parallel_env = Catch(
        num_drones=2,
        render_mode=None,
        init_flying_pos=jnp.array([[1, -1, 1], [1, 1, 1]]),
        init_target_location=jnp.array([0, 0, 1]),
        target_speed=0.1,
    )

    state, key = parallel_env.reset(key)

    assert (state.target_location > jnp.array([[-0.1, 0, 1]]) - 0.001).all()
    assert (state.target_location < jnp.array([[-0.1, 0, 1]]) + 0.001).all()

    parallel_env = Catch(
        num_drones=2,
        render_mode=None,
        init_flying_pos=jnp.array([[1, -1, 1], [1, 1, 1]]),
        init_target_location=jnp.array([1, 0, 1]),
        target_speed=0.1,
    )

    state, key = parallel_env.reset(key)

    assert jnp.linalg.norm(state.target_location - jnp.array([1, 0, 1])) <= 0.01
