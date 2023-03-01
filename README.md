[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# CrazyRL

:warning: Work in progress, suggestions are welcome. :warning:

Simple Particle Environment base on [Gymnasium](https://gymnasium.farama.org/) for mono agent and
[PettingZoo](https://pettingzoo.farama.org/) for parallel multiple agents.
The learning can be performed by SAC from stable baseline with [monoagent.py](learning/monoagent.py) for mono agent and
with [MASAC](https://github.com/ffelten/MASAC) for multiple agents.

Once the environment trained it can be displayed on simulation or in reality with [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/)
with the usage of [cflib](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/)

## Simulation

The simulation is a simple particle representation on a 3D cartesian reference base on Crazyflie [lighthouse reference frame](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/lighthouse/terminology_definitions/)

## Real

Positioning is managed by [Lighthouse positioning](https://www.bitcraze.io/documentation/system/positioning/ligthouse-positioning-system/).

### Guideline

Firstly configuration of the lighthouse has to be saved on config file. To do that you have to connect your Crazyflie
through the [cfclient app](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/),
manage the geometry for the lighthouse, estimate geometry simple and save the configuration on a yaml file.
Refer the path on [utils.py](utils/utils.py) on the load_config method and the configuration will be load on drones at each start up.

Secondly place the turned on drones on your environment. Be careful to put your drones at their right place depending on
their id to avoid any crash at start up.

#### multi agent
Then you can run the [test_multiagent](learning/test_multiagent.py) by notify the --total-timestep (100 should be a
correct amount for first test). The path to the save model masac and the mode to "real" has to be set on the main.

#### mono agent

Then you can run the [test_real_monoagent](learning/test_real_monoagent.py) by notify the save folder path with --save
and the --total-timesteps. The mode "simu" has to be set.


The drones should land on their starting position after the last action. I'm sorry there is not emergency landing
in case of exception on the code (should be a good things to add)

### Tips

If the drones are not starting once the test, verify your config load on the Crazyflie with the [cfclient](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/userguides/userguide_client/).
A method is launch at each start which reset the position estimator and load the config. If the information are not good
the drone will not take off.

Verify also the leds on drones if there is a continued red light it means the drone have not enough battery to pursue
the mission.

the led on lighthouse deck has to be green to ensure a good reception of lighthouse positioning
