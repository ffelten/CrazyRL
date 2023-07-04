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

🤝 Unified under a standard API from [PettingZoo](https://pettingzoo.farama.org/) parallel environments;

🚀 A JAX version faster on GPU;

✅ Good quality and documented Python code;

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

Execution :
```python
from crazy_rl.multi_agent.numpy.circle.circle import Circle

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

You can have a look at the [test_multiagent](learning/exec_multiagent.py) file. The path to the save model MASAC and the mode to "real" has to be set on the main.

### JAX version

This version is specifically optimized for GPU usage and intended for agent training purposes.
However, simulation and real-world functionalities are not available in this version.

Execution:

```python
from jax import random
from crazy_rl.multi_agent.jax.circle.circle import Circle

parallel_env = Circle(
        num_drones=5,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        num_intermediate_points=100,
    )

seed = 5  # test value
key = random.PRNGKey(seed)

key, subkey = random.split(key)
state = parallel_env.reset(subkey)

for i in range(500):
    while not (jnp.any(state.terminations) or jnp.any(state.truncations)):

        # where you would choose the drones' actions (2D array)

        key, subkey = random.split(key)
        state = parallel_env.step(state, actions, key)

        # where you would learn or add to buffer

    key, subkey = random.split(key)
    state = parallel_env.reset(subkey)
```

### Vmapped JAX version

The JAX version supports vmap, enabling parallelized training, particularly beneficial for GPU utilization.
While it offers faster performance on GPUs, it may exhibit slower execution on CPUs.

Execution:

```python
from jax import random, vmap
from crazy_rl.multi_agent.jax.circle.circle import Circle

parallel_env = Circle(
        num_drones=5,
        init_flying_pos=jnp.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 0.0, 1.0]]),
        num_intermediate_points=100,
    )

num_envs = 1000  # number of states in parallel
seed = 5  # test value
key = random.PRNGKey(seed)

vmapped_step = vmap(parallel_env.step_vmap)
vmapped_auto_reset = vmap(parallel_env.auto_reset)
vmapped_reset = vmap(parallel_env.reset)

key, *subkeys = random.split(key, num_envs + 1)
states = vmapped_reset(jnp.stack(subkeys))

for i in range(1000):

    # where you would choose the drones' actions (3D array)

    key, *subkeys = random.split(key, num_envs + 1)
    states = vmapped_step(actions, jnp.stack(subkeys), **parallel_env.state_to_dict(states))

    # where you would learn or add to buffer

    states = vmapped_auto_reset(**parallel_env.state_to_dict(states))
```

You can have a look to the main of [circle](crazy_rl/multi_agent/jax/circle/circle.py) for a jitted version with
the optimized 'fori_loop' or JAX.

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
`utils/` contains the basic functions to interact with the drones and OpenGL stuff for rendering.

You can explore the [test files](crazy_rl/test) to gain examples of usage and make comparisons between the
Numpy and JAX versions.

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
