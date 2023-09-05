"""Executing MAPPO policy in the real world."""
import argparse
import random
import time
from typing import Sequence

import cflib
import chex
import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from distrax import MultivariateNormalDiag
from etils import epath
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from pettingzoo import ParallelEnv

import crazy_rl.utils.geometry
from crazy_rl.multi_agent.numpy.catch import Catch  # noqa
from crazy_rl.multi_agent.numpy.circle import Circle  # noqa
from crazy_rl.multi_agent.numpy.escort import Escort  # noqa
from crazy_rl.multi_agent.numpy.surround import Surround  # noqa
from crazy_rl.utils.utils import LoggingCrazyflie


# NN from MAPPO
class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, local_obs_and_id: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(local_obs_and_id)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(actor_logtstd)
        pi: MultivariateNormalDiag = distrax.MultivariateNormalDiag(actor_mean, std)
        return pi


def _one_hot(agent_id: int, num_agents: int):
    return jnp.eye(num_agents)[agent_id]


def _norm_obs(obs: np.ndarray, min_obs: float, max_obs: float, low: float = -1.0, high: float = 1.0):
    return low + (obs - min_obs) * (high - low) / (max_obs - min_obs)


def _ma_get_action(actor: Actor, actor_state: TrainState, env: ParallelEnv, obs: dict, keys: chex.PRNGKey) -> dict:
    """Gets the action for all agents."""
    actions = {}
    for i, (key, value) in enumerate(obs.items()):
        # normalize obs
        normalized_obs = _norm_obs(value, min(env.observation_space(key).low), max(env.observation_space(key).high))
        # add agent id to obs
        agent_id = _one_hot(i, len(obs))
        agent_obs = jnp.concatenate([jnp.asarray(normalized_obs), agent_id])
        # get action from NN
        # print("Observation: ", agent_obs)
        pi = actor.apply(actor_state.params, agent_obs)
        action = pi.mode()  # deterministic mode just takes the mean
        # clip action
        action = jnp.clip(action, -1.0, 1.0)
        actions[key] = np.array(action)
    return actions


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


def play_episode(actor_module, actor_state, env, init_obs, key, simu):
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
    ep_return = 0.0
    while not done:
        # Execute policy for each agent
        print("Current obs: ", obs)
        start = time.time()
        key, subkey = jax.random.split(key)
        action_keys = jax.random.split(subkey, env.num_agents)
        actions = _ma_get_action(actor_module, actor_state, env, obs, action_keys)

        print("Time for model inference: ", time.time() - start)
        # print("Actions ", actions)

        start = time.time()
        next_obs, r, terminateds, truncateds, _ = env.step(actions)
        print("Time for env step: ", time.time() - start)
        ep_return += sum(r.values())

        if simu:
            time.sleep(0.05)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        done = terminated or truncated
        obs = next_obs
    print("==========Episode return: ", ep_return)


def load_actor_state(model_path: str, actor_state: TrainState):
    print(type(actor_state))
    directory = epath.Path(model_path)
    print("Loading actor from ", directory)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    actor_state = ckptr.restore(model_path, item=actor_state)
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

    # env = Circle(
    #     drone_ids=np.array([0, 1, 2]),
    #     render_mode="human",
    #     init_flying_pos=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
    # )

    env = Surround(
        drone_ids=np.arange(5),
        render_mode="human",
        init_flying_pos=np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.5, 1.5],
                [2.0, 0.5, 1.0],
                # [2.0, 2.5, 2.0],
                # [2.0, 1.0, 2.5],
                # [0.5, 0.5, 0.5],
            ]
        ),
        target_location=np.array([0.0, 0.5, 1.5]),
    )

    # env: ParallelEnv = Circle(
    #     drone_ids=np.array([0, 1, 2, 3, 4]),
    #     render_mode="human",
    #     init_flying_pos=np.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
    #     # target_location=np.array([1, 1, 2.5]),
    # )

    _ = env.reset(seed=args.seed)
    single_action_space = env.action_space(env.unwrapped.agents[0])
    key, actor_key = jax.random.split(key, 2)
    init_local_state = jnp.asarray(env.observation_space(env.unwrapped.agents[0]).sample())
    init_local_state_and_id = jnp.append(init_local_state, _one_hot(0, env.num_agents))  # add a fake id to init the actor net
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Use pretrained model
    actor_module = Actor(single_action_space.shape[0])
    actor_state = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_local_state_and_id),
        tx=optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=0.01, eps=1e-5),  # not used
        ),
    )
    actor_state = load_actor_state(args.model_dir, actor_state)

    obs, _ = env.reset(seed=args.seed)
    play_episode(actor_module, actor_state, env, obs, key, True)
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
    target_id = 3
    drones_ids = np.array([0, 1, 2])
    uris = ["radio://0/4/2M/E7E7E7E7" + str(id).zfill(2) for id in np.concatenate((drones_ids, [target_id]))]

    # Writes geometry to crazyflie
    for id in drones_ids:
        crazy_rl.utils.geometry.save_and_check("crazy_rl/utils/geometry.yaml", id, verbose=True)
    crazy_rl.utils.geometry.save_and_check("crazy_rl/utils/geometry.yaml", target_id, verbose=True)

    # the Swarm class will automatically launch the method in parameter of parallel_safe method
    with Swarm(uris, factory=CachedCfFactory(rw_cache="./cache")) as swarm:
        swarm.parallel_safe(LoggingCrazyflie)
        # swarm.reset_estimators()
        swarm.get_estimated_positions()

        # env: ParallelEnv = Circle(
        #     drone_ids=drones_ids,
        #     render_mode="real",
        #     init_flying_pos=np.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        #     # target_location=np.array([1, 1, 2.5]),
        #     swarm=swarm,
        #     # target_id=target_id,
        # )

        env = Surround(
            drone_ids=drones_ids,
            render_mode="real",
            init_flying_pos=np.array(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.5, 1.5],
                    [2.0, 0.5, 1.0],
                    # [2.0, 2.5, 2.0],
                    # [2.0, 1.0, 2.5],
                    # [0.5, 0.5, 0.5],
                ]
            ),
            target_location=np.array([0.0, 0.5, 1.5]),
            target_id=target_id,
        )

        obs, _ = env.reset(seed=args.seed)
        single_action_space = env.action_space(env.unwrapped.agents[0])
        key, actor_key = jax.random.split(key, 2)
        init_local_state = jnp.asarray(env.observation_space(env.unwrapped.agents[0]).sample())
        init_local_state_and_id = jnp.append(
            init_local_state, _one_hot(0, env.num_agents)
        )  # add a fake id to init the actor net
        assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Use pretrained model
        actor_module = Actor(single_action_space.shape[0])
        actor_state = TrainState.create(
            apply_fn=actor_module.apply,
            params=actor_module.init(actor_key, init_local_state_and_id),
            tx=optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(learning_rate=0.01, eps=1e-5),  # not used
            ),
        )
        actor_state = load_actor_state(args.model_dir, actor_state)

        play_episode(actor_module, actor_state, env, obs, key, False)

        env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "simu":
        replay_simu(args=args)
    elif args.mode == "real":
        replay_real(args=args)
