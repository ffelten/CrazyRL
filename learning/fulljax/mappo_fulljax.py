"""This heavily based on PPO from PureJaxRL https://github.com/luchris429/purejaxrl/tree/main/purejaxrl.

It is a super fast implementation of MAPPO, fully compiled on the GPU."""
import argparse
import os
import time
from distutils.util import strtobool
from typing import List, NamedTuple, Optional, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from distrax import MultivariateNormalDiag
from etils import epath
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import vmap

from crazy_rl.multi_agent.jax.base_parallel_env import State
from crazy_rl.multi_agent.jax.catch import Catch  # noqa
from crazy_rl.multi_agent.jax.circle import Circle  # noqa
from crazy_rl.multi_agent.jax.escort import Escort  # noqa
from crazy_rl.multi_agent.jax.surround import Surround  # noqa
from crazy_rl.utils.experiments_and_plots import save_results  # noqa
from crazy_rl.utils.jax_wrappers import (
    AddIDToObs,
    AutoReset,
    ClipActions,
    LogWrapper,
    NormalizeObservation,
    NormalizeVecReward,
    VecEnv,
)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="run in debug mode")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=128, help="the number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=10, help="the number of steps per epoch (higher batch size should be better)")
    parser.add_argument("--total-timesteps", type=int, default=3e6,
                        help="total timesteps of the experiments")
    parser.add_argument("--update-epochs", type=int, default=2, help="the number epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, default=2, help="the number of minibatches (keep small in MARL)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--gae-lambda", type=float, default=0.99,
                        help="the lambda for the generalized advantage estimation")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="the epsilon for clipping in the policy objective")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="the coefficient for the entropy bonus")
    parser.add_argument("--vf-coef", type=float, default=0.8,
                        help="the coefficient for the value function loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--activation", type=str, default="tanh",
                        help="the activation function for the neural networks")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to anneal the learning rate linearly")

    args = parser.parse_args()
    # fmt: on
    return args


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
        pi: MultivariateNormalDiag = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        return pi


class Critic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, global_obs: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(global_obs)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    terminated: jnp.ndarray
    joint_actions: jnp.ndarray  # shape is (num_envs, num_agents, action_dim)
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # shape is (num_envs, num_agents, obs_dim)
    global_obs: jnp.ndarray  # shape is (num_envs, global_obs_dim)
    info: jnp.ndarray


def make_train(args):
    num_updates = args.total_timesteps // args.num_steps // args.num_envs
    minibatch_size = args.num_envs * args.num_steps // args.num_minibatches

    def train(key: chex.PRNGKey, lr: Optional[float] = None):
        num_drones = 4
        env = Surround(
            num_drones=num_drones,
            init_flying_pos=jnp.array(
                [
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.5, 1.5],
                    [0.0, 0.5, 1.5],
                    # [0.5, 0.0, 1.0],
                    [0.5, -0.5, 1.5],
                    # [2.0, 2.5, 2.0],
                    # [2.0, 1.0, 2.5],
                    # [0.5, 0.5, 0.5],
                ]
            ),
            target_location=jnp.array([0.0, 0.3, 1.3]),
            # init_target_location=jnp.array([-1.0, -1.5, 2.0]),
            # final_target_location=jnp.array([1.0, 1.5, 1.0]),
        )
        # env = Circle(
        #     num_drones=num_drones,
        #     init_flying_pos=jnp.array([[-0.5, 0.0, 1.0], [0.0, 0.5, 1.0], [0.5, 0.0, 1.0]]),
        # )

        env = ClipActions(env)
        env = NormalizeObservation(env)
        env = AddIDToObs(env, num_drones)
        env = LogWrapper(env)
        env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
        env = VecEnv(env)  # vmaps the env public methods
        env = NormalizeVecReward(env, args.gamma)

        # Initial reset to have correct dimensions in the observations
        obs, info, state = env.reset(key=jax.random.PRNGKey(args.seed))

        lr = args.lr if lr is None else lr

        def linear_schedule(count):
            frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
            return lr * frac

        # INIT NETWORKS
        actor = Actor(env.action_space(0).shape[0], activation=args.activation)
        critic = Critic(activation=args.activation)
        key, actor_key, critic_key = jax.random.split(key, 3)
        dummy_local_obs_and_id = jnp.zeros(env.observation_space(0).shape[0] + num_drones)
        dummy_global_obs = jnp.zeros(env.state(state).shape)
        actor_params = actor.init(actor_key, dummy_local_obs_and_id)
        critic_params = critic.init(critic_key, dummy_global_obs)
        if args.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5),
            )

        actor_train_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=tx,
        )

        critic_train_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=tx,
        )

        def _ma_get_pi(params, obs: jnp.ndarray):
            """Gets the actions for all agents in all the envs at once. This is done with a for loop because distrax does not like vmapping."""
            return [actor.apply(params, obs[:, i, :]) for i in range(num_drones)]

        def _ma_sample_and_log_prob_from_pi(pi: List[MultivariateNormalDiag], key: chex.PRNGKey):
            """Samples actions for all agents in all the envs at once. This is done with a for loop because distrax does not like vmapping.

            Args:
                pi (List[MultivariateNormalDiag]): List of distrax distributions for agent actions (batched over envs)
                key (chex.PRNGKey): PRNGKey to use for sampling: size should be (num_agents, 2)
            """
            return [pi[i].sample_and_log_prob(seed=key[i]) for i in range(num_drones)]

        # Batch get value for parallel envs
        vmapped_get_value = vmap(critic.apply, in_axes=(None, 0))

        # INIT ENV
        key, subkeys = jax.random.split(key)
        reset_rngs = jax.random.split(subkeys, args.num_envs)
        obs, info, env_states = env.reset(reset_rngs)

        # TRAIN LOOP
        def _update_step(runner_state: Tuple[TrainState, TrainState, chex.Array, State, chex.PRNGKey], unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                actor_state, critic_state, obs, env_states, key = runner_state
                last_obs = obs

                # SELECT ACTION
                key, subkey = jax.random.split(key)
                # pi contains the normal distributions for each drones, batched (num_drones x  Distribution(num_envs, action_dim))
                pi = _ma_get_pi(actor_state.params, obs)
                action_keys = jax.random.split(subkey, num_drones)
                # for each env:
                #   for each agent:
                #       sample an action
                actions, log_probs = zip(*_ma_sample_and_log_prob_from_pi(pi, action_keys))
                actions = jnp.array(actions)
                log_probs = jnp.array(log_probs)
                log_probs = log_probs.transpose()  # (num_envs, num_drones) for storage
                joint_actions = actions.transpose((1, 0, 2))  # (num_envs, num_drones, action_dim) for storage

                # CRITIC STEP
                global_obss = env.state(env_states)
                values = vmapped_get_value(critic_state.params, global_obss)

                # STEP ENV
                key, subkey = jax.random.split(key)
                keys_step = jax.random.split(subkey, args.num_envs)
                obs, rewards, terminateds, truncateds, info, env_states = env.step(
                    env_states, joint_actions, jnp.stack(keys_step)
                )
                reward = rewards.sum(axis=-1)  # team reward
                terminated = jnp.logical_or(
                    jnp.any(terminateds, axis=-1), jnp.any(truncateds, axis=-1)
                )  # TODO handle truncations
                transition = Transition(
                    terminated=terminated,  # num_envs
                    joint_actions=joint_actions,  # (num_envs, num_drones, action_dim)
                    value=values,  # num_envs
                    reward=reward,  # num_envs
                    log_prob=log_probs,  # (num_envs, num_drones)
                    obs=last_obs,  # (num_envs, num_drones, obs_dim)
                    global_obs=global_obss,  # (num_envs, global_obs_dim)
                    info=info,  # dict containing fields of size (num_envs, ...)
                )

                runner_state = (actor_state, critic_state, obs, env_states, key)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, args.num_steps)

            # CALCULATE ADVANTAGE
            actor_train_state, critic_train_state, obs, env_states, key = runner_state
            global_obss = env.state(env_states)
            # TODO global_obss should be based on last obs, not current obs if truncated
            last_val = critic.apply(critic_train_state.params, global_obss)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.terminated,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + args.gamma * next_value * (1 - done) - value
                    gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(actor_critic_train_state, batch_info):
                    actor_train_state, critic_train_state = actor_critic_train_state
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(actor_params, critic_params, traj_batch, gae, targets):
                        # Batch values are in shape (batch_size, num_drones, ...)

                        # RERUN NETWORK
                        pi = _ma_get_pi(
                            actor_params, traj_batch.obs
                        )  # this is a list of distributions with batch_shape of minibatch_size and event shape of action_dim
                        new_value = vmapped_get_value(critic_params, traj_batch.global_obs)
                        # MA Log Prob: shape (num_drones, minibatch_size)
                        new_log_probs = jnp.array(
                            [pi[i].log_prob(traj_batch.joint_actions[:, i, :]) for i in range(num_drones)]
                        )
                        new_log_probs = new_log_probs.transpose()  # (minibatch_size, num_drones)

                        # Normalizes advantage (trick)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        gae = gae.reshape((-1, 1))  # (minibatch_size, 1)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (new_value - traj_batch.value).clip(
                            -args.clip_eps, args.clip_eps
                        )
                        value_losses = jnp.square(new_value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS FOR ALL AGENTS, AGGREGATE LOSS (sum)
                        logratio = new_log_probs - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        approx_kl = ((ratio - 1) - logratio).mean()
                        loss_actor1 = -ratio * gae
                        loss_actor2 = -jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * gae
                        loss_per_agent = jnp.maximum(loss_actor1, loss_actor2).mean(0)  # mean across minibatch
                        loss_actors = jnp.sum(loss_per_agent)  # sum across agents

                        entropies = jnp.array([p.entropy().mean() for p in pi])
                        entropy = entropies.mean()  # TODO check how to aggregate entropies

                        total_loss = loss_actors + args.vf_coef * value_loss - args.ent_coef * entropy
                        return total_loss, (value_loss, loss_actors, entropy, approx_kl)

                    grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                    total_loss_and_debug, grads = grad_fn(
                        actor_train_state.params, critic_train_state.params, traj_batch, advantages, targets
                    )
                    actor_train_state = actor_train_state.apply_gradients(grads=grads[0])
                    critic_train_state = critic_train_state.apply_gradients(grads=grads[1])
                    return (actor_train_state, critic_train_state), total_loss_and_debug

                actor_train_state, critic_train_state, traj_batch, advantages, targets, key = update_state
                key, subkey = jax.random.split(key)
                batch_size = minibatch_size * args.num_minibatches
                assert (
                    batch_size == args.num_steps * args.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(subkey, batch_size)
                batch = (traj_batch, advantages, targets)
                # flattens the num_steps and num_envs dimensions into batch_size; keeps the other dimensions untouched (num_drones, obs_dim, ...)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                # shuffles the full batch using permutations
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                # Slices the shuffled batch into num_minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [args.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                actor_critic_state, total_loss_and_debug = jax.lax.scan(
                    _update_minbatch, (actor_train_state, critic_train_state), minibatches
                )
                update_state = (actor_critic_state[0], actor_critic_state[1], traj_batch, advantages, targets, key)
                return update_state, total_loss_and_debug

            update_state = (actor_train_state, critic_train_state, traj_batch, advantages, targets, key)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, args.update_epochs)

            # Updates the train states (don't forget)
            actor_train_state = update_state[0]
            critic_train_state = update_state[1]
            key = update_state[-1]
            metric = traj_batch.info

            # Careful, metric has a shape of (num_steps, num_envs) and loss_info has a shape of (update_epochs, num_minibatches)
            losses = (
                loss_info[0],
                loss_info[1][0],
                loss_info[1][1],
                loss_info[1][2],
                loss_info[1][3],
            )
            # metric["total_loss"] = losses[0]
            # metric["value_loss"] = losses[1]
            # metric["actor_loss"] = losses[2]
            # metric["entropy"] = losses[3]
            # metric["approx_kl"] = losses[4]

            if args.debug:

                def callback(info, loss):
                    print(f"total loss: {loss[0].mean()}")
                    print(f"value loss: {loss[1].mean()}")
                    print(f"actor loss: {loss[2].mean()}")
                    print(f"entropy: {loss[3].mean()}")
                    print(f"approx kl: {loss[4].mean()}")
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    length = info["returned_episode_lengths"][info["returned_episode"]]
                    timesteps_when_done = info["timestep"][info["returned_episode"]] * args.num_envs
                    total_timesteps = info["total_timestep"][-1].sum()
                    if len(timesteps_when_done) > 0:
                        print(
                            f"global step={timesteps_when_done[0]}, episodic return={return_values.mean()}, length={length.mean()}"
                        )
                    print(f"==== total timesteps: {total_timesteps}")

                jax.debug.callback(callback, metric, losses)

            runner_state = (actor_train_state, critic_train_state, obs, env_states, key)
            return runner_state, metric

        key, subkey = jax.random.split(key)
        runner_state = (actor_train_state, critic_train_state, obs, env_states, subkey)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metric}

    return train


def save_actor(actor_state):
    directory = epath.Path("trained_model")
    actor_dir = directory / "actor_surround_4"
    print("Saving actor to ", actor_dir)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    ckptr.save(actor_dir, actor_state, force=True)


def multi_seeds(rng, args):
    NUM_SEEDS = 10
    rngs = jax.random.split(rng, NUM_SEEDS)
    train_vjit = jax.jit(jax.vmap(make_train(args), in_axes=(0,)))
    start_time = time.time()
    out = jax.block_until_ready(train_vjit(rngs, None))
    print(f"total time: {time.time() - start_time}")
    print(f"SPS: {args.total_timesteps * NUM_SEEDS / (time.time() - start_time)}")

    for i in range(NUM_SEEDS):
        returns = out["metrics"]["returned_episode_returns"][i]
        returns = returns.mean(-1).reshape(-1)
        returns = returns[returns != 0.0]  # filters out non-terminal returns
        plt.plot(returns)
    plt.show()


def hp_search(args):
    NUM_SEEDS = 3
    hyperparams = jnp.array([1e-4, 5e-4, 1e-3, 5e-3])  # lr

    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, NUM_SEEDS)
    train_vvjit = jax.jit(
        jax.vmap(
            jax.vmap(make_train(args), in_axes=(None, 0)),  # vmaps over the hyperparam
            in_axes=(0, None),  # vmaps over the rngs
        )
    )
    start_time = time.time()
    out = jax.block_until_ready(train_vvjit(rngs, hyperparams))
    print(f"total time: {time.time() - start_time}")
    print(f"SPS: {args.total_timesteps * NUM_SEEDS / (time.time() - start_time)}")

    import matplotlib.pyplot as plt

    for i in range(len(hyperparams)):
        returns = out["metrics"]["returned_episode_returns"][:, i, :]
        returns = returns.mean(0).mean(-1).reshape(-1)
        returns = returns[returns != 0.0]  # filters out non-terminal returns
        plt.plot(returns, label="lr=" + str(hyperparams[i]))


if __name__ == "__main__":
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    start_time = time.time()
    train_jit = jax.jit(make_train(args))  # one seed
    out = jax.block_until_ready(train_jit(rng, None))
    total_time = time.time() - start_time
    print(f"total time: {total_time}")
    print(f"SPS: {args.total_timesteps / total_time}")

    actor_state = out["runner_state"][0]
    save_actor(actor_state)

    import matplotlib.pyplot as plt

    returns = out["metrics"]["returned_episode_returns"]
    ep_length = out["metrics"]["returned_episode_lengths"]
    returns = returns.mean(-1)  # flattens seeds
    returns = returns.reshape(returns.shape[:1] + (-1,))  # flattens parallel envs
    returns = returns.reshape(-1)  # flattens rollouts
    returns = returns[returns != 0.0]  # filters out non-terminal returns

    return_with_timestep_and_time = []
    for i in range(len(returns)):
        timestep = (i + 1) * 200  # Circle episodes are always 200 timesteps (truncation)
        time = total_time / len(returns) * (i + 1)
        return_with_timestep_and_time.append(
            (
                timestep,
                time,  # assumes consistent SPS because it is impossible to get time inside jitted function
                returns[i],
            )
        )

    return_with_timestep_and_time = np.array(return_with_timestep_and_time)
    # save_results(return_with_timestep_and_time, f"MAPPO_GPU_Circle_({args.num_envs}envs)", args.seed)

    plt.plot(returns, label="episode return")

    # plt.plot(out["metrics"]["total_loss"].mean(-1).reshape(-1), label="total loss")
    # plt.plot(out["metrics"]["actor_loss"].mean(-1).reshape(-1), label="actor loss")
    # plt.plot(out["metrics"]["value_loss"].mean(-1).reshape(-1), label="value loss")
    # plt.plot(out["metrics"]["entropy"].mean(-1).reshape(-1), label="entropy")
    # plt.plot(out["metrics"]["approx_kl"].mean(-1).reshape(-1), label="approx kl")
    plt.xlabel("Update Step")
    plt.ylabel("")
    plt.legend()
    plt.show()
