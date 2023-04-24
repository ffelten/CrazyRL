<img src="swarm.gif" alt="Swarm" align="right" width="50%"/>

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![Test: pytest](https://github.com/ffelten/CrazyRL/actions/workflows/test.yml/badge.svg)

# CrazyRL

:warning: Work in progress, suggestions are welcome. :warning:

A library for doing Multi-Agent Reinforcement Learning with [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/) drones.

It is:

⚡️ A lightweight and fast simulator that is good enough to control [Crazyflies](https://www.bitcraze.io/products/crazyflie-2-1/) in practice;

🚁 A set of utilities based on the [cflib](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/) to control actual Crazyflies;

🤝 Unified under a standard API from [PettingZoo](https://pettingzoo.farama.org/) parallel environments;

✅ Good quality and documented Python code;

👷 A set of example environments to learn swarming behaviors (in progress).


The real-life example shown in the video is the result of executing the policies in real-life after learning in the lightweight simulator. The learning was performed by with [MASAC](https://github.com/ffelten/MASAC). Once the environment trained it can be displayed on simulation environment or in reality with the [Crazyflies](https://www.bitcraze.io/products/crazyflie-2-1/).

## Environments

The red balls represent the position of the drones.

### Hover

The drones learn to [hover](crazy_rl/multi_agent/hover/hover.py) in a fixed position.

<img src="hover.gif" alt="Hover" width="30%"/>

The yellow balls represent the target position of the drones.

### Circle
The drones learn to perform a coordinated [circle](crazy_rl/multi_agent/circle/circle.py).

<img src="circle.gif" alt="Circle" width="30%"/>

The yellow balls represent the target position of the drones.

### Surround
The drones learn to [surround](crazy_rl/multi_agent/surround/surround.py) a fixed target point.

<img src="surround.gif" alt="Surround" width="30%"/>

The yellow ball represents the target the drones have to surround.

### Escort
The drones learn to [escort](crazy_rl/multi_agent/escort/escort.py) a target moving straight to one point to another.

<img src="escort.gif" alt="Escort" width="30%"/>

The yellow ball represents the target the drones have to escort.

## API

### Training
I suggest to have a look at [MASAC](https://github.com/ffelten/MASAC) for training the agents.

### Execution
```python
    env: ParallelEnv = Circle(
        drone_ids=np.array([0, 1]),
        render_mode="human",    # or real, or None
        init_flying_pos=np.array([[0, 0, 1], [2, 2, 1]]),
    )

    done = False
    while not done:
        # Execute policy for each agent
        actions: Dict[str, np.ndarray] = {}
        for agent_id in env.possible_agents:
            obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
            act, _, _ = actor.get_action(obs_with_id.unsqueeze(0)) # YOUR POLICY HERE
            act = act.detach().cpu().numpy()
            actions[agent_id] = act.flatten()

        obs, _, _, _, _ = env.step(actions)
```

You can have a look at the [test_multiagent](learning/test_multiagent.py) file. The path to the save model MASAC and the mode to "real" has to be set on the main.

## Install & run
```shell
poetry install
poetry run python crazy_rl/multi_agent/circle/circle.py
```

## Simulation
`render_mode = "human"`

The simulation is a simple particle representation on a 3D cartesian reference based on Crazyflie [lighthouse reference frame](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/terminology_definitions/).
It is sufficient since the control of the CrazyFlies is high-level and precise enough.

## Real
`render_mode = "real"`

In our experiments, positioning was managed by [Lighthouse positioning](https://www.bitcraze.io/documentation/system/positioning/ligthouse-positioning-system/). It can probably be deployed with other positioning systems too.

### Guidelines

Firstly configuration of the positioning system has to be saved in a config file. The following explains quickly how to set up the LightHouse positioning system.

Then, connect your Crazyflie through the [cfclient app](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/), manage the geometry for the lighthouse, estimate geometry simple and save the configuration on a yaml file. You can then connect the other drones and load the geometry in them using the client.

(optional) Refer the path on [utils.py](crazy_rl/utils/utils.py) on the load_config method and the configuration will be load on drones at each start up. *This line has been commented out because it was very slow in practice. We just made sure the config was loaded before running the experiments.*

Secondly place the turned on drones on your environment, on the ground below the positions given to `init_flying_pos` in your code. Be careful to put your drones at their right place depending on their id to avoid any crash at start up.

### Tips

Verify also that the LEDs on drones aren't red: it means the drone have not enough battery to pursue the mission.

The LED on lighthouse deck have to be green to ensure a good reception of lighthouse positioning.


## Dev infos

### Structure
The switch between real environment and simulation is specified through the `render_mode` option, can be `"real"`, `"human"` or `None`.

`BaseParallelEnv` is the base class for the environment. It contains the basic methods to interact with the environment. From there, child classes allow to specify specific tasks such as Circle or Hover.
`utils/` contains the basic functions to interact with the drones and OpenGL stuff for rendering.

## Contributors
Pierre-Yves Houitte wrote the original version of this library. It has been cleaned up and simplified by Florian Felten (@ffelten).

## Citation
If you use this code for your research, please cite this using:

```bibtex
@misc{crazyrl,
    author = {Florian Felten and Pierre-Yves Houitte and El-Ghazali Talbi and Grégoire Danoy},
    title = {CrazyRL: A Multi-Agent Reinforcement Learning library for flying Crazyflie drones},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ffelten/CrazyRL}},
}
```
