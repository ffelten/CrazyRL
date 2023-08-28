"""File for testing the learned policies on the multiagent environment. Loads a Pytorch model and runs it on the environment."""
import argparse
import random
import time
from functools import partial
from typing import Dict

import cflib
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from etils import epath
from flax.training.train_state import TrainState
from pettingzoo import ParallelEnv

from crazy_rl.multi_agent.numpy.circle import Circle
from crazy_rl.multi_agent.numpy.surround import Surround
from crazy_rl.utils.utils import LoggingCrazyflie


# From MASAC Jax
def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


class Actor(nn.Module):
    """Actor Network for MASAC, it takes the local state of each agent and its id and returns a continuous action."""

    action_dim: int
    hidden_units: int = 256
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, local_obs_and_id: jnp.ndarray):
        # local state, id -> ... -> action_mean, action_std
        network = nn.Sequential(
            [
                nn.Dense(self.hidden_units, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.Dense(self.hidden_units, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.Dense(self.hidden_units, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
            ]
        )
        mean_layer = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)
        )
        log_std_layer = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        trunk = network(local_obs_and_id)
        mean, log_std = mean_layer(trunk), log_std_layer(trunk)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


@partial(jax.jit, static_argnames="actor_module")
def sample_action(
    actor_module: Actor,
    actor_state: TrainState,
    local_observations_and_ids: jnp.ndarray,
    key: jax.random.KeyArray,
) -> jnp.array:
    """Sample an action from the actor network then feed it to a gaussian distribution to get a continuous action.

    Args:
        actor_module: Actor network.
        actor_state: Actor network parameters.
        local_observations_and_ids: Local observations and agent ids.
        key: JAX random key.

    Returns: A tuple of (action, key).
    """
    key, subkey = jax.random.split(key, 2)
    mean, log_std = actor_module.apply(actor_state.params, local_observations_and_ids)
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
    action = jnp.tanh(gaussian_action)
    return action, key


def denormalize_action(action_space: gym.spaces.Box, scaled_action: np.ndarray) -> np.ndarray:
    """Rescale the action from [-1, 1] to [low, high]."""
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


@partial(jax.jit, static_argnames="actor_module")
def sample_action_for_agent(actor_module: Actor, actor_state: TrainState, obs: jnp.ndarray, agent_id, key):
    """Samples an action from the policy given an observation and an agent_id. This is vmapped to get all actions at each timestep."""
    obs_with_ids = jnp.append(obs[agent_id], agent_id)
    act, key = sample_action(actor_module, actor_state, obs_with_ids, key)
    return act, key


def parse_args():
    """Parse the arguments from the command line."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--model-dir", type=str, required=True, help="the dir of the model to load.")

    parser.add_argument("--mode", type=str, default="simu", choices=["simu", "real"],
                        help="choose the replay mode to perform real or simulation")
    args = parser.parse_args()
    # fmt: on
    return args


def play_episode(actor_module, actor_state, env, init_obs, single_action_space, key, simu):
    """Play one episode.

    Args:
        actor_module: the actor network
        actor_state: the actor network parameters
        env: the environment
        init_obs: initial observations
        single_action_space: the action space of a single agent
        key: the random key
        simu: true if simulation, false if real
    """
    obs = init_obs
    done = False
    agent_ids = jnp.arange(env.num_agents)
    while not done:
        # Execute policy for each agent
        actions: Dict[str, np.ndarray] = {}
        print("Current obs: ", obs)
        start = time.time()
        obs_array = jnp.array([obs[agent] for agent in env.agents])
        acts, keys = jax.vmap(sample_action_for_agent, in_axes=(None, None, None, 0, None))(
            actor_module, actor_state, obs_array, agent_ids, key
        )
        # TODO split into subkeys?
        key = keys[0]

        # Construct the dict of actions for PZ
        for agent_id, act in zip(env.agents, acts):
            act = np.array(act)
            # Clip due to numerical instability
            act = np.clip(act, -1, 1)
            # Rescale to proper domain when using squashing
            act = denormalize_action(single_action_space, act)
            actions[agent_id] = act

        print("Time for model inference: ", time.time() - start)

        # TRY NOT TO MODIFY: execute the game and log data.
        start = time.time()
        next_obs, r, terminateds, truncateds, infos = env.step(actions)
        print(r)
        print("Time for env step: ", time.time() - start)

        if simu:
            time.sleep(0.05)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        done = terminated or truncated
        obs = next_obs


def load_actor_state(model_path: str, actor_state: TrainState):
    print(type(actor_state))
    directory = epath.Path(model_path)
    print("Loading actor from ", directory)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    ckptr.restore(model_path, item=actor_state)
    print(type(actor_state))

    return actor_state


def replay_simu(args):
    """Replay the simulation for one episode.

    Args:
        args: the arguments from the command line
    """

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    env: ParallelEnv = Circle(
        drone_ids=np.array([0, 1, 2]),
        render_mode="human",
        init_flying_pos=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        # init_flying_pos=np.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        # target_location=np.array([1, 1, 2.5]),
    )

    _ = env.reset(seed=args.seed)
    single_action_space = env.action_space(env.unwrapped.agents[0])
    key, actor_key = jax.random.split(key, 2)
    init_local_state = jnp.asarray(env.observation_space(env.unwrapped.agents[0]).sample())
    init_local_state_and_id = jnp.append(init_local_state, jnp.array([0]))  # add a fake id to init the actor net
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Use pretrained model
    actor_module = Actor(action_dim=np.prod(single_action_space.shape))
    actor_state = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_local_state_and_id),
        tx=optax.adam(learning_rate=0.1),  # not used
    )
    actor_state = load_actor_state(args.model_dir, actor_state)

    for i in range(10):
        obs, _ = env.reset(seed=args.seed)
        play_episode(actor_module, actor_state, env, obs, single_action_space, key, True)
    env.close()


def replay_real(args):
    """Replay the real world for one episode.

    Args:
        args: the arguments from the command line
    """
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Init swarm config of crazyflie
    cflib.crtp.init_drivers()
    uris = {
        "radio://0/4/2M/E7E7E7E700",
        "radio://0/4/2M/E7E7E7E701",
        # Add more URIs if you want more copters in the swarm
    }
    # uri = 'radio://0/4/2M/E7E7E7E7' + str(id).zfill(2) # you can browse the drone_id and add as this code at the end of the uri

    # the Swarm class will automatically launch the method in parameter of parallel_safe method
    factory = CachedCfFactory(rw_cache="./cache")

    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(LoggingCrazyflie)
        # swarm.reset_estimators()
        swarm.get_estimated_positions()

        env: ParallelEnv = Surround(
            drone_ids=np.array([0, 1, 2, 3, 4]),
            render_mode="real",
            init_flying_pos=np.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
            target_location=np.array([1, 1, 2.5]),
            swarm=swarm,
        )

        obs, _ = env.reset(seed=args.seed)
        single_action_space = env.action_space(env.unwrapped.agents[0])
        key, actor_key = jax.random.split(key, 2)
        init_local_state = jnp.asarray(env.observation_space(env.unwrapped.agents[0]).sample())
        init_local_state_and_id = jnp.append(init_local_state, jnp.array([0]))  # add a fake id to init the actor net
        assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Use pretrained model
        actor_module = Actor(action_dim=np.prod(single_action_space.shape))
        actor_state = TrainState.create(
            apply_fn=actor_module.apply,
            params=actor_module.init(actor_key, init_local_state_and_id),
            tx=optax.adam(learning_rate=0.1),  # not used
        )
        actor_state = load_actor_state(args.model_dir, actor_state)

        play_episode(actor_module, actor_state, env, obs, single_action_space, key, True)

        env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "simu":
        replay_simu(args=args)
    elif args.mode == "real":
        replay_real(args=args)
