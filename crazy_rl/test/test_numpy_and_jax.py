import jax.numpy as jnp
import jax.random as random
import numpy as np

from crazy_rl.multi_agent.jax.surround.surround import Surround as JaxSurround
from crazy_rl.multi_agent.numpy.surround.surround import Surround as NpSurround


def compare(state, np_env, observations, terminations, truncations, rewards):
    assert (state.agents_locations[0] == np_env._agent_location["agent_0"]).all()
    assert (state.agents_locations[1] == np_env._agent_location["agent_1"]).all()

    assert state.timestep == np_env.timestep

    assert (state.observations[0] == observations["agent_0"]).all()
    assert (state.observations[1] == observations["agent_1"]).all()

    assert state.terminations[0] == terminations["agent_0"]
    assert state.terminations[1] == terminations["agent_1"]

    assert state.truncations[0] == truncations["agent_0"]
    assert state.truncations[1] == truncations["agent_1"]

    assert state.rewards[0] > rewards["agent_0"] - 0.01
    assert state.rewards[0] < rewards["agent_0"] + 0.01

    assert state.rewards[1] > rewards["agent_1"] - 0.01
    assert state.rewards[1] < rewards["agent_1"] + 0.01


def test_np_jax():
    """Test to compare numpy and jax API."""
    # Jax initialisation

    jax_env = JaxSurround(
        num_drones=2,
        init_flying_pos=jnp.array([[0, 0, 1], [0, 1, 1]]),
        target_location=jnp.array([[1, 0, 1]]),
        size=2,
    )

    seed = 5
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    state = jax_env.reset(subkey)

    # Numpy initialisation

    np_env = NpSurround(
        drone_ids=np.array([0, 1]),
        render_mode=None,
        init_flying_pos=np.array([[0, 0, 1], [0, 1, 1]]),
        target_location=np.array([1, 0, 1]),
        size=2,
    )

    observations, infos = np_env.reset(seed=seed)

    # Fist action

    np_actions = {"agent_0": np.array([0, 1, 0]), "agent_1": np.array([0, -1, 0])}  # [0, 0.2, 1] [0, 0.8, 1]
    observations, rewards, terminations, truncations, infos = np_env.step(np_actions)

    jax_actions = jnp.array([[0, 1, 0], [0, -1, 0]])
    key, subkey = random.split(key)
    state = jax_env.step(state, jax_actions, subkey)

    compare(state, np_env, observations, terminations, truncations, rewards)

    # Collision

    observations, rewards, terminations, truncations, infos = np_env.step(np_actions)  # [0, 0.4, 1] [0, 0.6, 1]

    key, subkey = random.split(key)
    state = jax_env.step(state, jax_actions, subkey)

    np_actions = {"agent_0": np.array([0, 0.5, 0]), "agent_1": np.array([0, 0, 0])}  # [0, 0.5, 1] [0, 0.6, 1]
    observations, rewards, terminations, truncations, infos = np_env.step(np_actions)

    jax_actions = jnp.array([[0, 0.5, 0], [0, 0, 0]])
    key, subkey = random.split(key)
    state = jax_env.step(state, jax_actions, subkey)

    compare(state, np_env, observations, terminations, truncations, rewards)

    assert (state.rewards == jnp.array([-10, -10])).all()

    assert state.terminations.all()

    key, subkey = random.split(key)
    state = jax_env.reset(subkey)
    observations, infos = np_env.reset()

    # Border

    np_actions = {"agent_0": np.array([0, 0, 0]), "agent_1": np.array([0, 1, 0])}

    jax_actions = jnp.array([[0, 0, 0], [0, 1, 0]])

    for i in range(5):
        observations, rewards, terminations, truncations, infos = np_env.step(np_actions)

        key, subkey = random.split(key)
        state = jax_env.step(state, jax_actions, subkey)

    compare(state, np_env, observations, terminations, truncations, rewards)

    # End

    np_actions = {"agent_0": np.array([0, 0, 0]), "agent_1": np.array([0, 0, 0])}
    jax_actions = jnp.array([[0, 0, 0], [0, 0, 0]])

    for i in range(95):
        observations, rewards, terminations, truncations, infos = np_env.step(np_actions)
        key, subkey = random.split(key)
        state = jax_env.step(state, jax_actions, subkey)

    compare(state, np_env, observations, terminations, truncations, rewards)

    assert state.truncations.all()

    key, subkey = random.split(key)
    state = jax_env.reset(subkey)
    observations, infos = np_env.reset()

    assert (state.observations[0] == observations["agent_0"]).all()
    assert (state.observations[1] == observations["agent_1"]).all()
