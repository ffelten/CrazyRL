"""File for testing the learned policies on the multiagent environment. Loads a Pytorch model and runs it on the environment."""
import argparse
import random
import time
from distutils.util import strtobool
from typing import Dict

import cflib
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID

from crazy_rl.multi_agent.surround import Surround
from crazy_rl.utils.utils import LoggingCrazyflie


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Actor network for the multiagent environment. From MASAC."""

    def __init__(self, env: ParallelEnv):
        """Initialize the actor network."""
        super().__init__()
        single_action_space = env.action_space(env.agents[0])
        single_observation_space = env.observation_space(env.agents[0])
        # Local state, agent id -> ... -> local action
        self.fc1 = nn.Linear(np.array(single_observation_space.shape).prod() + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((single_action_space.high - single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((single_action_space.high + single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        """Forward pass of the actor network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        """Get an action from the actor network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def parse_args():
    """Parse the arguments from the command line."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--model-filename", type=str, required=True, help="the filename of the model to load.")

    parser.add_argument("--mode", type=str, default="simu", choices=["simu", "real"],
                        help="choose the replay mode to perform real or simulation")
    args = parser.parse_args()
    # fmt: on
    return args


def concat_id(local_obs: np.ndarray, id: AgentID) -> np.ndarray:
    """Concatenate the agent id to the local observation.

    Args:
        local_obs: the local observation
        id: the agent id to concatenate

    Returns: the concatenated observation

    """
    return np.concatenate([local_obs, np.array([extract_agent_id(id)], dtype=np.float32)])


def extract_agent_id(agent_str):
    """Extract agent id from agent string.

    Args:
        agent_str: Agent string in the format of "agent_{id}"

    Returns: (int) Agent id

    """
    return int(agent_str.split("_")[1])


def play_episode(actor, env, init_obs, device):
    """Play one episode.

    Args:
        actor: the actor network
        env: the environment
        args: the arguments
        device: the device to use
    """
    obs = init_obs
    done = False
    while not done:
        # Execute policy for each agent
        actions: Dict[str, np.ndarray] = {}
        print("Current obs: ", obs)
        start = time.time()
        with torch.no_grad():
            for agent_id in env.possible_agents:
                obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
                act, _, _ = actor.get_action(obs_with_id.unsqueeze(0))
                act = act.detach().cpu().numpy()
                actions[agent_id] = act.flatten()
        print("Time for model inference: ", time.time() - start)

        # TRY NOT TO MODIFY: execute the game and log data.
        start = time.time()
        next_obs, _, terminateds, truncateds, infos = env.step(actions)
        print("Time for env step: ", time.time() - start)

        time.sleep(0.2)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        done = terminated or truncated
        obs = next_obs


def replay_simu(args):
    """Replay the simulation for one episode.

    Args:
        args: the arguments from the command line
    """

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print("Using ", device)

    env: ParallelEnv = Surround(
        drone_ids=np.array([0, 1, 2, 3, 4]),
        render_mode="human",
        init_flying_pos=np.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
        target_location=np.array([1, 1, 2.5]),
    )

    obs = env.reset(seed=args.seed)
    single_action_space = env.action_space(env.unwrapped.agents[0])
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Use pretrained model
    actor = Actor(env).to(device)
    if args.model_filename is not None:
        print("Loading pre-trained model ", args.model_filename)
        actor.load_state_dict(torch.load(args.model_filename))
        actor.eval()

    play_episode(actor, env, obs, device)
    env.close()


def replay_real(args):
    """Replay the real world for one episode.

    Args:
        args: the arguments from the command line
    """
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print("Using ", device)

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
    if args.model_filename is not None:
        print("Loading pre-trained model ", args.model_filename)
        model_params = torch.load(args.model_filename)
    else:
        raise ValueError("Please specify the model filename to load.")
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

        obs = env.reset(seed=args.seed)
        # Use pretrained model
        print("Loading pre-trained model ", args.model_filename)
        actor = Actor(env).to(device)
        actor.load_state_dict(model_params)
        actor.eval()
        print("Model loaded. Starting to play episode.")

        play_episode(actor, env, obs, device)

        env.close()


if __name__ == "__main__":

    # time.sleep(5)

    args = parse_args()

    if args.mode == "simu":
        replay_simu(args=args)
    elif args.mode == "real":
        replay_real(args=args)
