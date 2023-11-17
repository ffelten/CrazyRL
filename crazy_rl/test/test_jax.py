import jax.numpy as jnp
import jax.random as random
import numpy as np

from crazy_rl.multi_agent.jax.catch.catch import Catch
from crazy_rl.multi_agent.jax.circle.circle import Circle
from crazy_rl.multi_agent.jax.escort.escort import Escort
from crazy_rl.multi_agent.jax.surround.surround import Surround


def two_drones_crash(parallel_env, key):
    """Test where two drones starting on (0, 0, 1), (0, 1, 1) crash together."""
    obs, info, state = parallel_env.reset(key)

    obs, rewards, terminated, truncated, info, state, key = move(parallel_env, state, key, jnp.array([[0, 1, 0], [0, -1, 0]]))

    assert (state.agents_locations == jnp.array([[0, 0.2, 1], [0, 0.8, 1]])).all()

    obs, rewards, terminated, truncated, info, state, key = move(parallel_env, state, key, jnp.array([[0, 1, 0], [0, -1, 0]]))

    assert not terminated.any()

    obs, rewards, terminated, truncated, info, state, key = move(parallel_env, state, key, jnp.array([[0, 1, 0], [0, -1, 0]]))

    # the drones crash
    assert terminated.all()

    assert (rewards == jnp.array([-10, -10])).all()

    return obs, rewards, terminated, truncated, state


def crash_ground(parallel_env, key):
    """Test where the first drone, starting on (0, 0, 1), crashes on the ground."""
    obs, info, state = parallel_env.reset(key)

    obs, rewards, terminated, truncated, info, state, key = move(
        parallel_env, state, key, jnp.array([[0, 0, -1], [0, 0, 0]]), 4
    )  # position = [0, 0, 0.2] (0.20003)

    obs, rewards, terminated, truncated, info, state, key = move(
        parallel_env, state, key, jnp.array([[0, 0, -0.01], [0, 0, 0]])
    )  # position = [0, 0, 0.2] (just below)

    # the drone crashes on the ground
    assert terminated.all()

    return obs, rewards, terminated, truncated, state


def leave_the_map(parallel_env, key, total_moves=200):
    """Test where the second drone, starting on (0, 1, 1) tries to leave the map, and they stay alive until the
    end of the round."""
    obs, info, state = parallel_env.reset(key)

    obs, rewards, terminated, truncated, info, state, key = move(
        parallel_env, state, key, jnp.array([[0, 0, 0], [0, 1, 0]]), 10
    )

    # the drone stays in the map
    assert state.agents_locations[1, 1] <= 3

    obs, rewards, terminated, truncated, info, state, key = wait(parallel_env, state, key, total_moves - 10)

    # the game ends after 100 timesteps
    assert truncated.all()

    return obs, rewards, terminated, truncated, state


def move(parallel_env, state, key, actions, iterations=1):
    for i in range(iterations):
        key, subkey = random.split(key)
        obs, rewards, terminated, truncated, info, state = parallel_env.step(state, actions, subkey)
    return obs, rewards, terminated, truncated, info, state, key


def wait(parallel_env, state, key, iterations=1):
    return move(parallel_env, state, key, jnp.array([[0, 0, 0], [0, 0, 0]]), iterations)


def test_surround():
    """Test for the Surround environment in jax version."""
    parallel_env = Surround(
        num_drones=2,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
        size=3,
    )

    seed = 5

    key = random.PRNGKey(seed)

    # 1st round : the two drones crash

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = two_drones_crash(parallel_env, subkey)

    # 2nd round : one drone crashes with the target

    key, subkey = random.split(key)
    obs, info, state = parallel_env.reset(subkey)

    obs, rewards, terminated, truncated, info, state, key = move(
        parallel_env, state, key, jnp.array([[0, 0, 0], [1, 0, 1]])
    )  # position = [0.2, 1, 1.2]

    assert (obs == jnp.array([[0, 0, 1, 1, 1, 2.5, 0.2, 1, 1.2], [0.2, 1, 1.2, 1, 1, 2.5, 0, 0, 1]])).all()

    obs, rewards, terminated, truncated, info, state, key = move(
        parallel_env, state, key, jnp.array([[0, 0, 0], [1, 0, 1]]), 4
    )  # position = [1, 1, 2]

    obs, rewards, terminated, truncated, info, state, key = move(
        parallel_env, state, key, jnp.array([[0, 0, 0], [0, 0, 1]]), 2
    )  # position = [1, 1, 2.4]

    # the drone crashes in the target
    assert terminated.all()

    # 3rd round : one drone crashes with the ground

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = crash_ground(parallel_env, subkey)

    # 4th round : the drones never crash and one tries to leave the map

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = leave_the_map(parallel_env, subkey)


def test_circle():
    """Test for the Circle environment in jax version."""
    parallel_env = Circle(
        num_drones=2,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        num_intermediate_points=100,
        size=3,
    )

    seed = 5

    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = leave_the_map(parallel_env, subkey)

    # observation : agent's location and target's location
    assert (obs == jnp.array([[0, 0, 1, -0.5, 0, 1], [0, 3, 1, -0.5, 1, 1]])).all()

    # assert (rewards > jnp.array([-0.5, -jnp.linalg.norm(jnp.array([0.5, 2, 0]))]) - 0.01).all()
    # assert (rewards < jnp.array([-0.5, -jnp.linalg.norm(jnp.array([0.5, 2, 0]))]) + 0.01).all()

    key, subkey = random.split(key)
    obs, info, state = parallel_env.reset(subkey)

    obs, rewards, terminated, truncated, info, state, key = wait(parallel_env, state, key)

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
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        init_target_location=jnp.array([1, 1, 2.5]),
        final_target_location=jnp.array([-2, -2, 3]),
        num_intermediate_points=150,
        size=3,
    )

    seed = 5

    key = random.PRNGKey(seed)

    # 1st round : the two drones crash
    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = two_drones_crash(parallel_env, subkey)

    # 2nd round : one drone crashes with the ground

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = crash_ground(parallel_env, subkey)

    # 3rd round : the drones never crash and one tries to leave the map

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = leave_the_map(parallel_env, subkey)

    # observation : agent's location, target's location and other agent's location
    assert (
        obs
        == jnp.array(
            [
                [
                    0,
                    0,
                    1,
                    state.target_location[0, 0],
                    state.target_location[0, 1],
                    state.target_location[0, 2],
                    0,
                    3,
                    1,
                ],
                [
                    0,
                    3,
                    1,
                    state.target_location[0, 0],
                    state.target_location[0, 1],
                    state.target_location[0, 2],
                    0,
                    0,
                    1,
                ],
            ]
        )
    ).all()

    key, subkey = random.split(key)
    obs, info, state = parallel_env.reset(subkey)

    obs, rewards, terminated, truncated, info, state, key = wait(parallel_env, state, key)

    assert (state.target_location == jnp.array([1, 1, 2.5]) + jnp.array([-3, -3, 0.5]) / 152).all()


def test_catch():
    """Test for the Catch environment in jax version."""
    parallel_env = Catch(
        num_drones=2,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        init_target_location=jnp.array([1, 1, 2.5]),
        target_speed=0.1,
        size=3,
    )

    seed = 5

    key = random.PRNGKey(seed)

    # 1st round : the two drones crash

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = two_drones_crash(parallel_env, subkey)

    # 2nd round : one drone crashes with the ground

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = crash_ground(parallel_env, subkey)

    # 3rd round : the drones never crash and one tries to leave the map

    key, subkey = random.split(key)
    obs, rewards, terminated, truncated, state = leave_the_map(parallel_env, subkey)

    # observation : agent's location, target's location and other agent's location
    assert (
        obs
        == jnp.array(
            [
                [
                    0,
                    0,
                    1,
                    state.target_location[0, 0],
                    state.target_location[0, 1],
                    state.target_location[0, 2],
                    0,
                    3,
                    1,
                ],
                [
                    0,
                    3,
                    1,
                    state.target_location[0, 0],
                    state.target_location[0, 1],
                    state.target_location[0, 2],
                    0,
                    0,
                    1,
                ],
            ]
        )
    ).all()

    # Tests the target

    parallel_env = Catch(
        num_drones=2,
        init_flying_pos=jnp.array([[1, -1, 1], [1, 1, 1]]),
        init_target_location=jnp.array([0, 0, 1]),
        target_speed=0.1,
    )

    key, subkey = random.split(key)
    obs, info, state = parallel_env.reset(subkey)

    obs, rewards, terminated, truncated, info, state, key = wait(parallel_env, state, key)

    assert (state.target_location > jnp.array([[-0.1, 0, 1]]) - 0.001).all()
    assert (state.target_location < jnp.array([[-0.1, 0, 1]]) + 0.001).all()

    parallel_env = Catch(
        num_drones=2,
        init_flying_pos=jnp.array([[1, -1, 1], [1, 1, 1]]),
        init_target_location=jnp.array([1, 0, 1]),
        target_speed=0.1,
    )

    key, subkey = random.split(subkey)
    obs, info, state = parallel_env.reset(key)

    obs, rewards, terminated, truncated, info, state, key = wait(parallel_env, state, key)

    assert jnp.linalg.norm(state.target_location - jnp.array([1, 0, 1])) <= 0.01


def test_action_space():
    """Test the sampling of action_space."""
    parallel_env = Surround(
        num_drones=2,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
    )

    seed = 5

    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    obs, info, state = parallel_env.reset(subkey)

    actions = jnp.array([parallel_env.action_space(agent).sample(subkey) for agent in range(parallel_env.num_drones)])

    assert parallel_env.action_space(0).contains(actions[0])
    assert parallel_env.action_space(0).contains(actions[1])

    obs, rewards, terminated, truncated, info, state = parallel_env.step(state, actions, key)

    actions = jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=jnp.float32)

    assert parallel_env.action_space(0).contains(actions[0])
    assert parallel_env.action_space(0).contains(actions[1])

    actions = np.array([[1.0, 2.0, 30.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)

    assert not (parallel_env.action_space(0).contains(actions))


def test_observation():
    """Test if the observation correspond to the observation space."""
    parallel_env = Surround(
        num_drones=2,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        target_location=jnp.array([[1, 1, 2.5]]),
    )

    seed = 5

    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    obs, info, state = parallel_env.reset(subkey)

    for agent in range(parallel_env.num_drones):
        assert parallel_env.observation_space(agent).contains(obs[agent])

    obs, rewards, terminated, truncated, info, state, key = move(parallel_env, state, key, jnp.array([[0, 0, 1], [0, 0, 1]]))

    for agent in range(parallel_env.num_drones):
        assert parallel_env.observation_space(agent).contains(obs[agent])
