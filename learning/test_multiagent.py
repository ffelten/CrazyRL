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

from crazy_rl.multi_agent.circle import Circle
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
    parser.add_argument("--model-filename", type=str, default="../../MASAC/masac.pt", help="the filename of the model to load.")

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

    env: ParallelEnv = Circle(
        drone_ids=[0, 1],
        render_mode="human",
        init_xyzs=[[0, 0, 0], [1, 1, 0]],
        init_target_points=[[0, 0, 1], [1, 1, 1]],
    )

    single_action_space = env.action_space(env.unwrapped.agents[0])
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Use pretrained model
    actor = Actor(env).to(device)
    if args.model_filename is not None:
        print("Loading pre-trained model ", args.model_filename)
        actor = torch.load(args.model_filename)

    # TRY NOT TO MODIFY: start the game
    obs: Dict[str, np.ndarray] = env.reset(seed=args.seed)
    done = False
    while not done:
        # Execute policy for each agent
        actions: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for agent_id in env.possible_agents:
                obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
                act, _, _ = actor.get_action(obs_with_id.unsqueeze(0))
                act = act.detach().cpu().numpy()
                actions[agent_id] = act.flatten()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, _, terminateds, truncateds, infos = env.step(actions)
        time.sleep(0.02)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        done = terminated or truncated
        obs = next_obs

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
    with Swarm(uris, factory=factory) as swarm:

        swarm.parallel_safe(LoggingCrazyflie)
        swarm.get_estimated_positions()

        env: ParallelEnv = Circle(
            render_mode="real",
            drone_ids=[0, 1],
            init_xyzs=[[0, 0, 0], [1, 1, 0]],
            init_target_points=[[0, 0, 1], [1, 1, 1]],
            swarm=swarm,
        )

        single_action_space = env.action_space(env.unwrapped.agents[0])
        assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

        # Use pretrained model
        actor = Actor(env).to(device)
        if args.model_filename is not None:
            print("Loading pre-trained model ", args.model_filename)
            actor = torch.load(args.model_filename)

        # TRY NOT TO MODIFY: start the game
        obs: Dict[str, np.ndarray] = env.reset(seed=args.seed)
        done = False
        while not done:
            # Exec policy for each agent
            actions: Dict[str, np.ndarray] = {}
            with torch.no_grad():
                for agent_id in env.possible_agents:
                    obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
                    act, _, _ = actor.get_action(obs_with_id.unsqueeze(0))
                    act = act.detach().cpu().numpy()
                    actions[agent_id] = act.flatten()

            next_obs, _, terminateds, truncateds, infos = env.step(actions)
            time.sleep(0.02)

            terminated: bool = any(terminateds.values())
            truncated: bool = any(truncateds.values())

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            done = terminated or truncated

        env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "simu":
        replay_simu(args=args)
    elif args.mode == "real":
        replay_real(args=args)
