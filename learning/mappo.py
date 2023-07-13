"""This heavily based on PPO from PureJaxRL https://github.com/luchris429/purejaxrl/tree/main/purejaxrl."""
import argparse
import os
import time
from distutils.util import strtobool
from typing import List, NamedTuple, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from distrax import MultivariateNormalDiag
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import vmap

from crazy_rl.multi_agent.jax.catch.catch import Catch, State
from crazy_rl.utils.jax_wrappers import (
    AddIDToObs,
    AutoReset,
    LogWrapper,
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
    parser.add_argument("--num-envs", type=int, default=2048, help="the number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=10, help="the number of steps per epoch (higher batch size should be better)")
    parser.add_argument("--total-timesteps", type=int, default=1e7,
                        help="total timesteps of the experiments")
    parser.add_argument("--update-epochs", type=int, default=4, help="the number epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, default=32, help="the number of minibatches (keep small in MARL)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the generalized advantage estimation")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="the epsilon for clipping in the policy objective")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="the coefficient for the entropy bonus")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="the coefficient for the value function loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--activation", type=str, default="tanh",
                        help="the activation function for the neural networks")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to anneal the learning rate linearly")
    parser.add_argument("--normalize-env", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to normalize the observations and rewards")

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
    joint_actions: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(args):
    num_updates = args.total_timesteps // args.num_steps // args.num_envs
    minibatch_size = args.num_envs * args.num_steps // args.num_minibatches
    num_drones = 3

    env = Catch(
        num_drones=num_drones,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        init_target_location=jnp.array([1.0, 1.0, 2.5]),
        target_speed=0.1,
    )

    env = AddIDToObs(env, num_drones)
    env = LogWrapper(env)
    env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
    env = VecEnv(env)  # vmaps the env public methods
    if args.normalize_env:
        # env = NormalizeObservation(env)
        env = NormalizeVecReward(env, args.gamma)

    # Initial reset to have correct dimensions in the observations
    obs, info, state = env.reset(key=jax.random.PRNGKey(0))

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
        return args.lr * frac

    def train(key: chex.PRNGKey):
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

        def _ma_get_pi(params, obs: jnp.ndarray):
            """Gets the actions for all agents in all the envs at once. This is done with a for loop because distrax does not like vmapping."""
            return [actor.apply(params, obs[:, i, :]) for i in range(num_drones)]

        def _ma_sample_and_log_prob_from_pi(pi: List[MultivariateNormalDiag], key: chex.PRNGKey):
            """Samples actions for all agents in all the envs at once. This is done with a for loop because distrax does not like vmapping.

            Args:
                pi (List[MultivariateNormalDiag]): List of distrax distributions for agent actions (batched over envs)
                key (chex.PRNGKey): PRNGKey to use for sampling: size should be (num_agents, 2)
            """
            assert key.shape == (num_drones, 2)
            return [pi[i].sample_and_log_prob(seed=key[i]) for i in range(num_drones)]

        critic_train_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=tx,
        )

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
                env_states: State
                actor_state, critic_state, obs, env_states, key = runner_state
                last_obs = obs

                # SELECT ACTION
                key, subkey = jax.random.split(key)
                # pi contains the normal distributions for each drones, batched (num_drones x  Distribution(num_envs, action_dim))
                pi = _ma_get_pi(actor_state.params, obs)
                action_keys = jax.random.split(subkey, num_drones)
                action_keys = action_keys.reshape((num_drones, -1))
                # for each env:
                #   for each agent:
                #       sample an action
                actions, log_probs = zip(*_ma_sample_and_log_prob_from_pi(pi, action_keys))
                joint_actions = jnp.array(actions).reshape((args.num_envs, num_drones, -1))
                log_probs = jnp.array(log_probs).reshape((args.num_envs, num_drones))  # TODO maybe should sum this?
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
                terminated = jnp.any(terminateds, axis=-1)  # team terminated
                # TODO check if terminated signal is correct
                transition = Transition(terminated, joint_actions, values, reward, log_probs, last_obs, global_obss, info)
                runner_state = (actor_state, critic_state, obs, env_states, key)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, args.num_steps)

            # CALCULATE ADVANTAGE
            actor_train_state, critic_train_state, obs, env_states, key = runner_state
            global_obss = env.state(env_states)
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
                        # RERUN NETWORK
                        pi = _ma_get_pi(actor_params, traj_batch.obs)
                        value = vmapped_get_value(critic_params, traj_batch.global_obs)
                        # MA Log Prob
                        log_probs = [pi[i].log_prob(traj_batch.joint_actions[:, i]) for i in range(num_drones)]
                        log_probs = jnp.array(log_probs).sum(axis=-1)
                        # .sum(axis=0)  # TODO check if should not aggregate logprobs across agents

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        def _actor_loss(log_prob, i):
                            ratio = jnp.exp(log_prob - traj_batch.log_prob[:, i])
                            loss_actor1 = ratio * gae
                            loss_actor2 = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * gae
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            return loss_actor

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-args.clip_eps, args.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS FOR ALL AGENTS
                        loss_actor = jax.vmap(_actor_loss, in_axes=(0, 0))(log_probs, jnp.arange(num_drones)).sum()
                        entropy = jnp.array([p.entropy().mean() for p in pi]).mean()  # TODO check how to aggregate entropies

                        total_loss = loss_actor + args.vf_coef * value_loss - args.ent_coef * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                    total_loss, grads = grad_fn(
                        actor_train_state.params, critic_train_state.params, traj_batch, advantages, targets
                    )
                    actor_train_state = actor_train_state.apply_gradients(grads=grads[0])
                    critic_train_state = critic_train_state.apply_gradients(grads=grads[1])
                    return (actor_train_state, critic_train_state), total_loss

                actor_train_state, critic_train_state, traj_batch, advantages, targets, key = update_state
                key, subkey = jax.random.split(key)
                batch_size = minibatch_size * args.num_minibatches
                assert (
                    batch_size == args.num_steps * args.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(subkey, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [args.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                actor_critic_state, total_loss = jax.lax.scan(
                    _update_minbatch, (actor_train_state, critic_train_state), minibatches
                )
                update_state = (actor_critic_state[0], actor_critic_state[1], traj_batch, advantages, targets, key)
                return update_state, total_loss

            update_state = (actor_train_state, critic_train_state, traj_batch, advantages, targets, key)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, args.update_epochs)
            metric = traj_batch.info
            key = update_state[-1]
            if args.debug:

                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * args.num_envs
                    total_timesteps = info["total_timestep"].sum()
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                    print(f"total timesteps: {total_timesteps}")

                jax.debug.callback(callback, metric)

            runner_state = (actor_train_state, critic_train_state, obs, env_states, key)
            return runner_state, metric

        key, subkey = jax.random.split(key)
        runner_state = (actor_train_state, critic_train_state, obs, env_states, subkey)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    args = parse_args()
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(args))
    start_time = time.time()
    out = jax.block_until_ready(train_jit(rng))
    print(f"total time: {time.time() - start_time}")
    print(f"SPS: {args.total_timesteps / (time.time() - start_time)}")

    import matplotlib.pyplot as plt

    plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.show()
