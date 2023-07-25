<img src="swarm.gif" alt="Swarm" align="right" width="50%"/>

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![Test: pytest](https://github.com/ffelten/CrazyRL/actions/workflows/test.yml/badge.svg)

# CrazyRL

:warning: Work in progress, suggestions are welcome. :warning:

A library for doing Multi-Agent Reinforcement Learning with [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/) drones.

It has:

⚡️ A lightweight and fast simulator that is good enough to control [Crazyflies](https://www.bitcraze.io/products/crazyflie-2-1/) in practice;

🚁 A set of utilities based on the [cflib](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/) to control actual Crazyflies;

🤝 A Numpy version, unified under a standard API from [PettingZoo](https://pettingzoo.farama.org/) parallel environments;

🚀 A [JAX](https://github.com/google/jax) version that can be run fully on GPU;

✅ Good quality, tested and documented Python code;

👷 A set of example environments to learn swarming behaviors (in progress).


The real-life example shown in the video is the result of executing the policies in real-life after learning in the lightweight simulator. The learning was performed by with [MASAC](https://github.com/ffelten/MASAC). Once the environment trained it can be displayed on simulation environment or in reality with the [Crazyflies](https://www.bitcraze.io/products/crazyflie-2-1/).

## Environments

The red balls represent the position of the drones.

### Hover

The drones learn to hover in a fixed position.

<img src="hover.gif" alt="Hover" width="30%"/>

The yellow balls represent the target position of the drones.

Available in [Numpy](crazy_rl/multi_agent/numpy/hover/hover.py) and [JAX](crazy_rl/multi_agent/jax/hover/hover.py)
version.

### Circle
The drones learn to perform a coordinated circle.

<img src="circle.gif" alt="Circle" width="30%"/>

The yellow balls represent the target position of the drones.

Available in [Numpy](crazy_rl/multi_agent/numpy/circle/circle.py) and [JAX](crazy_rl/multi_agent/jax/circle/circle.py)
version.

### Surround
The drones learn to surround a fixed target point.

<img src="surround.gif" alt="Surround" width="30%"/>

The yellow ball represents the target the drones have to surround.

Available in [Numpy](crazy_rl/multi_agent/numpy/surround/surround.py) and
[JAX](crazy_rl/multi_agent/jax/surround/surround.py) version.

### Escort
The drones learn to escort a target moving straight to one point to another.

<img src="escort.gif" alt="Escort" width="30%"/>

The yellow ball represents the target the drones have to surround.

Available in [Numpy](crazy_rl/multi_agent/numpy/escort/escort.py) and [JAX](crazy_rl/multi_agent/jax/escort/escort.py)
version.

### Catch
The drones learn to catch a target trying to escape.

<img src="catch.gif" alt="Catch" width="30%"/>

The yellow ball represents the target the drones have to surround.

Available in [Numpy](crazy_rl/multi_agent/numpy/catch/catch.py) and [JAX](crazy_rl/multi_agent/jax/catch/catch.py)
version.

## API

### Training
I suggest to have a look at [MASAC](https://github.com/ffelten/MASAC) for training the agents.

### Numpy version

Basic version which can be used for training, simulation and the real drones.
It follows the [PettingZoo parallel API](https://pettingzoo.farama.org/).

Execution :
```python
from crazy_rl.multi_agent.numpy.circle.circle import Circle

env: ParallelEnv = Circle(
    drone_ids=np.array([0, 1]),
    render_mode="human",    # or real, or None
    init_flying_pos=np.array([[0, 0, 1], [2, 2, 1]]),
)

obs, info = env.reset()

done = False
while not done:
    # Execute policy for each agent
    actions: Dict[str, np.ndarray] = {}
    for agent_id in env.possible_agents:
        actions[agent_id] = actor.get_action(obs[agent_id], agent_id)

    obs, _, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
```

You can have a look at the `learning/` folder to see how we execute pre-trained policies
using [MASAC](https://github.com/ffelten/MASAC) in both Torch and Jax.

### JAX version

This version is specifically optimized for GPU usage and intended for agent training purposes.
However, simulation and real-world functionalities are not available in this version.

Moreover, it is not compliant with the PettingZoo API as it heavily relies on functional programming.
We sacrificed the API compatibility for huge performance gains.

Some functionalities are automatically done by wrappers, such as vmap, enabling parallelized training, allowing to leverage
all the cores on the GPU.
While it offers faster performance on GPUs, it may exhibit slower execution on CPUs.

You can find other wrappers you may need defined in [jax_wrappers](crazy_rl/utils/jax_wrappers.py).

Execution:

```python
from jax import random
from crazy_rl.multi_agent.jax.circle.circle import Circle

parallel_env = Circle(
        num_drones=5,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        num_intermediate_points=100,
    )

num_envs = 3  # number of states in parallel
seed = 5  # test value
key = random.PRNGKey(seed)
key, *subkeys = random.split(key, num_envs + 1)

# Wrappers
env = AutoReset(env)  # Auto reset the env when done, stores additional info in the dict
env = VecEnv(env)  # vmaps the env public methods

obs, info, state = env.reset(jnp.stack(subkeys))

for i in range(301):

    actions = jnp.zeros((num_envs, parallel_env.num_drones, parallel_env.action_space(0).shape[0]))
    for env_id, obs in enumerate(states.observations):
        for agent_id in range(parallel_env.num_drones):
            actions[env_id, agent_id] = actor.get_action(obs, agent_id, key) # YOUR POLICY HERE

    key, *subkeys = random.split(key, num_envs + 1)
    obs, rewards, term, trunc, info, state = env.step(state, actions, jnp.stack(subkeys))

    # where you would learn or add to buffer
```

## Install & run

### Numpy version
```shell
poetry install
poetry run python crazy_rl/multi_agent/numpy/circle/circle.py
```

### JAX on CPU

```shell
poetry install
poetry run python crazy_rl/multi_agent/jax/circle/circle.py
```

### JAX on GPU

JAX GPU support is not included in the [pyproject.toml](pyproject.toml) file, as JAX CPU is the default option.
Therefore, you need to manually install JAX GPU and disregard the poetry requirements for this purpose.

```shell
poetry install
pip install --upgrade pip

# Using CUDA 12
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or using CUDA 11
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

poetry run python crazy_rl/multi_agent/jax/circle/circle.py
```

Please refer to the [JAX installation GitHub page](https://github.com/google/jax#installation) for the
specific CUDA version requirements.

After installation, the JAX version automatically utilizes the GPU as the default device. However, if you
prefer to switch to the CPU without reinstalling, you can manually set the device using the following command:

```python
jax.config.update("jax_platform_name", "cpu")
```

## Simulation

`render_mode = "human"`

The simulation is a simple particle representation on a 3D cartesian reference based on Crazyflie [lighthouse reference frame](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/terminology_definitions/).
It is sufficient since the control of the CrazyFlies is high-level and precise enough.

Available in the Numpy version.

## Real

`render_mode = "real"`

In our experiments, positioning was managed by [Lighthouse positioning](https://www.bitcraze.io/documentation/system/positioning/ligthouse-positioning-system/).
It can probably be deployed with other positioning systems too.

Available in the Numpy version.

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

The project consists of two versions, each with corresponding files located in the
[JAX directory](crazy_rl/multi_agent/jax) and the [Numpy directory](crazy_rl/multi_agent/numpy), respectively.

In the Numpy version, the switch between real environment and simulation is specified through the `render_mode`
option, can be `"real"`, `"human"` or `None`.

`BaseParallelEnv` is the base class for the environment in both versions. It contains the basic methods to
interact with the environment. From there, child classes allow to specify specific tasks such as Circle or Hover.
`utils/` contains the basic functions to interact with the drones, OpenGL stuff for rendering and wrappers which
add automatic behaviours to JAX version.

You can explore the [test files](crazy_rl/test) to gain examples of usage and make comparisons between the
Numpy and JAX versions.

## Contributors
* Florian Felten made the design, architecture, vision, reviews and cleanup.
* Pierre-Yves Houitte wrote an original proof-of-concept of the library.
* Coline Ledez adapted the environments to Jax, added tests.
* El-Ghazali Talbi and Grégoire Danoy supervised the work.

## Citation
If you use this code for your research, please cite this using:

```bibtex
@misc{crazyrl,
    author = {Florian Felten and Coline Ledez and Pierre-Yves Houitte and El-Ghazali Talbi and Grégoire Danoy},
    title = {CrazyRL: A Multi-Agent Reinforcement Learning library for flying Crazyflie drones},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ffelten/CrazyRL}},
}
```
