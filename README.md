[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# CrazyRL
:warning: Work in progress, suggestions are welcome. :warning:

A library for doing RL with [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/) drones. It contains a [PettingZoo](https://pettingzoo.farama.org/) environment for parallel multiple agents.
The learning can be performed by with [MASAC](https://github.com/ffelten/MASAC) for multiple agents.

Once the environment trained it can be displayed on simulation environment or in reality with [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/)
with the usage of [cflib](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/).

## API

### Training
I suggest to have a look at [MASAC](https://github.com/ffelten/MASAC) for training the agents.

### Execution
```python
    env: ParallelEnv = Circle(
        drone_ids=[0, 1],
        render_mode="human", # or real, or None
        init_xyzs=[[0, 0, 0], [1, 1, 0]],
        init_target_points=[[0, 0, 1], [1, 1, 1]],
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

        # TRY NOT TO MODIFY: execute the game and log data.
        obs, _, terminateds, truncateds, infos = env.step(actions)
```

You can have a look at the [test_multiagent](learning/test_multiagent.py) file. The path to the save model MASAC and the mode to "real" has to be set on the main.

## Simulation
`render_mode = "human"`

The simulation is a simple particle representation on a 3D cartesian reference based on Crazyflie [lighthouse reference frame](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/terminology_definitions/).
It is sufficient since the control of the CrazyFlies is quite high level already.

## Real
`render_mode = "real"`

Positioning is managed by [Lighthouse positioning](https://www.bitcraze.io/documentation/system/positioning/ligthouse-positioning-system/).

### Guideline

Firstly configuration of the lighthouse has to be saved on config file. To do that you have to connect your Crazyflie
through the [cfclient app](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/),
manage the geometry for the lighthouse, estimate geometry simple and save the configuration on a yaml file.
Refer the path on [utils.py](utils/utils.py) on the load_config method and the configuration will be load on drones at each start up.

Secondly place the turned on drones on your environment. Be careful to put your drones at their right place depending on
their id to avoid any crash at start up.

### Tips

If the drones are not starting once the test, verify your config load on the Crazyflie with the [cfclient](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/).
A method is launched at each start which reset the position estimator and load the config. If the information are not good
the drone will not take off.

Verify also the LEDs on drones if there is a continued red light it means the drone have not enough battery to pursue
the mission.

The LED on lighthouse deck have to be green to ensure a good reception of lighthouse positioning.


## Dev infos

### Structure
The switch between real environment and simulation is specified through the `render_mode` option, can be `"real"`, `"human"` or `None`.

`BaseParallelEnv` is the base class for the environment. It contains the basic methods to interact with the environment. From there, child classes allow to specify specific tasks such as Circle or Hover.
`utils/` contains the basic functions to interact with the drones and OpenGL stuff for rendering.
