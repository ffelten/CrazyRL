import jax.numpy as jnp
from jax import random

from crazy_rl.multi_agent.jax.catch.catch import Catch
from crazy_rl.utils.jax_wrappers import (
    AddIDToObs,
    AutoReset,
    LogWrapper,
    NormalizeObservation,
    NormalizeVecReward,
    VecEnv,
)


def is_normalized(rewards, obs):
    assert (rewards <= 1).all()
    assert (rewards >= -1.1).all()

    assert (obs <= 1).all()
    assert (obs >= -1).all()


def id_obs(obs):
    assert obs[0][0][-2] == 1  # one more dimension because of parallelization
    assert obs[0][0][-1] == 0
    assert obs[0][1][-2] == 0
    assert obs[0][1][-1] == 1


def test_normalize():
    num_agents = 2
    env = Catch(
        num_drones=num_agents,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        init_target_location=jnp.array([1.0, 1.0, 2.5]),
        target_speed=0.1,
    )

    num_envs = 3  # number of states in parallel
    seed = 5  # test value
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, num_envs + 1)

    # Wrappers
    env = NormalizeObservation(env)
    env = AddIDToObs(
        env, num_agents
    )  # concats the agent id as one hot encoded vector to the obs (easier for learning algorithms)
    env = LogWrapper(env)  # Add stuff in the info dictionary for logging in the learning algo
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods
    env = NormalizeVecReward(env, gamma=0.99)  # normalize the reward in [-1, 1]

    obs, info, state = env.reset(jnp.stack(subkeys))

    id_obs(obs)

    assert (obs <= 1).all()
    assert (obs >= -1).all()

    # The two drones crash

    actions = jnp.array([[[0, 1, 0], [0, -1, 0]], [[0, 1, 0], [0, -1, 0]], [[0, 0, -1], [0, 0, 0]]])

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))  # 0.2, 0.8

    id_obs(obs)

    is_normalized(rewards, obs)

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))  # 0.4, 0.6

    actions = jnp.array([[[0, 0.5, 0], [0, 0, 0]], [[0, 0.5, 0], [0, 0, 0]], [[0, 0.5, 0], [0, 0, 0]]])

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    is_normalized(rewards, obs)

    actions = jnp.array([[[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 1]]])

    for i in range(100):
        key, *subkeys = random.split(key, num_envs + 1)
        obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    is_normalized(rewards, obs)
