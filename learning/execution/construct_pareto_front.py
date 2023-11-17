"""Executing MAPPO policy in the real world."""
import argparse
import random
from typing import Sequence

import chex
import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint
from distrax import MultivariateNormalDiag
from etils import epath
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from mplcursors import cursor
from pettingzoo import ParallelEnv

# from crazy_rl.multi_agent.numpy.escort import Escort
from crazy_rl.multi_agent.numpy.surround import Surround
from crazy_rl.utils.pareto import ParetoArchive


# from crazy_rl.multi_agent.numpy.catch import Catch
# from crazy_rl.multi_agent.numpy.circle import Circle


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
    parser.add_argument("--models-dir", type=str, required=True, help="the dir of the model to load.")
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
    ep_return = np.zeros(2)
    while not done:
        # Execute policy for each agent
        key, subkey = jax.random.split(key)
        action_keys = jax.random.split(subkey, env.num_agents)
        actions = _ma_get_action(actor_module, actor_state, env, obs, action_keys)

        next_obs, r, terminateds, truncateds, _ = env.step(actions)
        ep_return += np.array(list(r.values())).sum(axis=0)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        done = terminated or truncated
        obs = next_obs
    return ep_return


def load_actor_state(model_path, actor_state: TrainState):
    directory = epath.Path(model_path)
    print("Loading actor from ", directory)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    actor_state = ckptr.restore(model_path, item=actor_state)

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
        drone_ids=np.arange(8),
        init_flying_pos=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 2.0, 2.0],
                [2.0, 0.5, 1.0],
                [2.0, 2.5, 2.0],
                [2.0, 1.0, 2.5],
                [0.5, 0.5, 0.5],
            ]
        ),
        target_location=np.array([1.0, 1.0, 2.0]),
        multi_obj=True,
        size=5,
    )

    # env = Catch(
    #     drone_ids=np.arange(8),
    #     render_mode="human",
    #     init_flying_pos=np.array(
    #         [
    #             [0.0, 0.0, 1.0],
    #             [0.0, 1.0, 1.0],
    #             [1.0, 0.0, 1.0],
    #             [1.0, 2.0, 2.0],
    #             [2.0, 0.5, 1.0],
    #             [2.0, 2.5, 2.0],
    #             [2.0, 1.0, 2.5],
    #             [0.5, 0.5, 0.5],
    #         ]
    #     ),
    #     init_target_location=np.array([1.0, 1.0, 2.0]),
    #     target_speed=0.15,
    #     # final_target_location=np.array([-2.0, -2.0, 1.0]),
    #     # num_intermediate_points=100,
    # )

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

    p = epath.Path(args.models_dir)
    model_dirs = [f for f in p.iterdir() if f.is_dir()]
    pareto_front = ParetoArchive()
    for model_dir in model_dirs:
        actor_state = load_actor_state(model_dir, actor_state)
        obs, _ = env.reset(seed=args.seed)
        policy_eval = play_episode(actor_module, actor_state, env, obs, key, True)
        pareto_front.add(candidate=model_dir, evaluation=policy_eval)

    env.close()
    return pareto_front


if __name__ == "__main__":
    args = parse_args()
    p = epath.Path(args.models_dir)
    model_dirs = [f for f in p.iterdir() if f.is_dir()]
    print(model_dirs)

    pf = replay_simu(args=args)

    for candidate, eval in zip(pf.individuals, pf.evaluations):
        plt.scatter(eval[0], eval[1], label=candidate.name, alpha=0.9, c="#5CB5FF")

    plt.ylabel("Far from others", fontsize=16)
    plt.xlabel("Close to target", fontsize=16)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("results/mo/pareto_front.png", dpi=600)
    plt.savefig("results/mo/pareto_front.pdf", dpi=600)
    cursor(hover=True)
    plt.show()
