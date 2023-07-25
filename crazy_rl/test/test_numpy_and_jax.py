import jax.numpy as jnp
import jax.random as random
import numpy as np

from crazy_rl.multi_agent.jax.surround.surround import Surround as JaxSurround
from crazy_rl.multi_agent.numpy.surround.surround import Surround as NpSurround


def compare(state, jax_obs, jax_term, jax_trunc, jax_rewards, np_env, np_obs, np_term, np_trunc, np_rewards):
    assert (state.agents_locations[0] == np_env._agent_location["agent_0"]).all()
    assert (state.agents_locations[1] == np_env._agent_location["agent_1"]).all()

    assert state.timestep == np_env.timestep

    assert (jax_obs[0] == np_obs["agent_0"]).all()
    assert (jax_obs[1] == np_obs["agent_1"]).all()

    assert jax_term[0] == np_term["agent_0"]
    assert jax_term[1] == np_term["agent_1"]

    assert jax_trunc[0] == np_trunc["agent_0"]
    assert jax_trunc[1] == np_trunc["agent_1"]

    assert jax_rewards[0] > np_rewards["agent_0"] - 0.01
    assert jax_rewards[0] < np_rewards["agent_0"] + 0.01

    assert jax_rewards[1] > np_rewards["agent_1"] - 0.01
    assert jax_rewards[1] < np_rewards["agent_1"] + 0.01


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
    jax_obs, info, state = jax_env.reset(subkey)

    # Numpy initialisation

    np_env = NpSurround(
        drone_ids=np.array([0, 1]),
        render_mode=None,
        init_flying_pos=np.array([[0, 0, 1], [0, 1, 1]]),
        target_location=np.array([1, 0, 1]),
        size=2,
    )

    np_obs, infos = np_env.reset(seed=seed)

    # Fist action

    np_actions = {"agent_0": np.array([0, 1, 0]), "agent_1": np.array([0, -1, 0])}  # [0, 0.2, 1] [0, 0.8, 1]
    np_obs, np_rewards, np_term, np_trunc, infos = np_env.step(np_actions)

    jax_actions = jnp.array([[0, 1, 0], [0, -1, 0]])
    key, subkey = random.split(key)
    jax_obs, jax_rewards, jax_term, jax_trunc, info, state = jax_env.step(state, jax_actions, subkey)

    compare(state, jax_obs, jax_term, jax_trunc, jax_rewards, np_env, np_obs, np_term, np_trunc, np_rewards)

    # Collision

    np_obs, np_rewards, np_term, np_trunc, infos = np_env.step(np_actions)  # [0, 0.4, 1] [0, 0.6, 1]

    key, subkey = random.split(key)
    jax_obs, jax_rewards, jax_term, jax_trunc, info, state = jax_env.step(state, jax_actions, subkey)

    np_actions = {"agent_0": np.array([0, 0.5, 0]), "agent_1": np.array([0, 0, 0])}  # [0, 0.5, 1] [0, 0.6, 1]
    np_obs, np_rewards, np_term, np_trunc, infos = np_env.step(np_actions)

    jax_actions = jnp.array([[0, 0.5, 0], [0, 0, 0]])
    key, subkey = random.split(key)
    jax_obs, jax_rewards, jax_term, jax_trunc, info, state = jax_env.step(state, jax_actions, subkey)

    compare(state, jax_obs, jax_term, jax_trunc, jax_rewards, np_env, np_obs, np_term, np_trunc, np_rewards)

    assert (jax_rewards == jnp.array([-10, -10])).all()

    assert jax_term.all()

    key, subkey = random.split(key)
    jax_obs, info, state = jax_env.reset(subkey)
    np_obs, infos = np_env.reset()

    # Border

    np_actions = {"agent_0": np.array([0, 0, 0]), "agent_1": np.array([0, 1, 0])}

    jax_actions = jnp.array([[0, 0, 0], [0, 1, 0]])

    for i in range(5):
        np_obs, np_rewards, np_term, np_trunc, infos = np_env.step(np_actions)

        key, subkey = random.split(key)
        jax_obs, jax_rewards, jax_term, jax_trunc, info, state = jax_env.step(state, jax_actions, subkey)

    compare(state, jax_obs, jax_term, jax_trunc, jax_rewards, np_env, np_obs, np_term, np_trunc, np_rewards)

    # End

    np_actions = {"agent_0": np.array([0, 0, 0]), "agent_1": np.array([0, 0, 0])}
    jax_actions = jnp.array([[0, 0, 0], [0, 0, 0]])

    for i in range(95):
        np_obs, np_rewards, np_term, np_trunc, infos = np_env.step(np_actions)
        key, subkey = random.split(key)
        jax_obs, jax_rewards, jax_term, jax_trunc, info, state = jax_env.step(state, jax_actions, subkey)

    compare(state, jax_obs, jax_term, jax_trunc, jax_rewards, np_env, np_obs, np_term, np_trunc, np_rewards)

    assert jax_trunc.all()

    key, subkey = random.split(key)
    jax_obs, info, state = jax_env.reset(subkey)
    np_obs, infos = np_env.reset()

    assert (jax_obs[0] == np_obs["agent_0"]).all()
    assert (jax_obs[1] == np_obs["agent_1"]).all()
