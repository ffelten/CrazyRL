"""Utils for the project."""
import time

import numpy as np
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger


def rad2deg(x):
    r"""Converts radians to degrees."""
    return 180 * x / np.pi


def deg2rad(x):
    r"""Converts degrees to radians."""
    return np.pi * x / 180


def reset_estimator(scf):
    """Reset the position estimation of the drone.

    Args:
        scf: SyncCrazyflie
    """
    cf = scf.cf
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    __wait_for_position_estimator(scf)


def __wait_for_position_estimator(scf):
    """Wait the position estimation and never end if is not good to block any action on drones id bad configuration.

    Args:
        scf: SyncCrazyflie
    """
    log_config = LogConfig(name="Kalman Variance", period_in_ms=500)
    log_config.add_variable("kalman.varPX", "float")
    log_config.add_variable("kalman.varPY", "float")
    log_config.add_variable("kalman.varPZ", "float")

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data["kalman.varPX"])
            var_x_history.pop(0)
            var_y_history.append(data["kalman.varPY"])
            var_y_history.pop(0)
            var_z_history.append(data["kalman.varPZ"])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            if (max_x - min_x) < threshold and (max_y - min_y) < threshold and (max_z - min_z) < threshold:
                break


def _activate_high_level_commander(scf):
    scf.cf.param.set_value("commander.enHighLevel", "1")


def run_take_off(scf):
    """Take off manoeuvre use in swarm classes with parallel_safe() method (each crazyflie of the swarm will launch it in parallel).

    Args:
        scf: SyncCrazyflie
    """
    # print("Setting controller")
    # scf.cf.param.set_value("stabilizer.controller", value=1)  # to use Mellinger controller set value to 2
    # print("controller activated")

    print("Taking off")
    commander = scf.cf.high_level_commander

    commander.takeoff(0.5, 2.0)
    time.sleep(2)


def run_square(scf):
    """Square manoeuvre use in swarm classes with parallel_safe() method (each crazyflie of the swarm will launch it in parallel)."""
    print("square")
    commander = scf.cf.high_level_commander
    commander.go_to(0.0, 1.0, 0.5, 0, 2.0)
    time.sleep(2)
    commander.go_to(1.0, 1.0, 0.5, 0, 2.0)
    time.sleep(2)
    commander.go_to(1.0, 0.0, 0.5, 0, 2.0)
    time.sleep(2)
    commander.go_to(0.0, 0.0, 0.5, 0, 2.0)
    time.sleep(2)


def run_land(scf):
    """Land manoeuvre use in swarm classes with parallel_safe() method (each crazyflie of the swarm will launch it in parallel).

    Args:
        scf: SyncCrazyflie
    """
    print("land")

    commander = scf.cf.high_level_commander

    commander.land(0.0, 2.0)
    time.sleep(2)

    commander.stop()

    scf.close_link()


def run_sequence(scf, command):
    """Method use in swarm classes with parallel_safe() method to go to the targeted point for each crazyflie  of the swarm in parallel. parallel_safe() method takes care of sequencing the dictionary of args by URI, thus command is each value in the dict.

    Args:
        scf: SyncCrazyFlie object given automatically by the Swarm class
        command: ndarray (2,3)-shaped array which include the current agent position and the target position (x,y,z)
    """
    agent_pos = command[0]
    target_pos = command[1]

    yaw = 0
    flight_time = np.linalg.norm(target_pos - agent_pos) * 1
    print(f"Move {agent_pos} -> {target_pos} ({flight_time} secs)")

    if flight_time == 0.0:
        print("first reset")
    else:
        commander = scf.cf.high_level_commander
        # z limitation for safety
        z = np.clip(target_pos[2], 0.3, 2.5)
        commander.go_to(target_pos[0], target_pos[1], z, yaw, flight_time, relative=False)
        time.sleep(flight_time * 0.65)


class LoggingCrazyflie:
    """Simple logging class that logs the Stabilizer from a supplied Crazyflie.

    TODO make it work with swarm class to allow access to more data than SwarmPosition()
    """

    def __init__(self, scf):
        """Initialize and run the example with the specified link_uri."""
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.rpy = np.zeros(3)

        self._lg_esti = LogConfig(name="Estimate", period_in_ms=10)
        self._lg_esti.add_variable("stateEstimate.x", "float")
        self._lg_esti.add_variable("stateEstimate.y", "float")
        self._lg_esti.add_variable("stateEstimate.z", "float")
        self._lg_esti.add_variable("stateEstimate.vx", "float")
        self._lg_esti.add_variable("stateEstimate.vy", "float")
        self._lg_esti.add_variable("stateEstimate.vz", "float")

        self.lg_stab_gyro = LogConfig(name="stab_gyro", period_in_ms=10)
        self.lg_stab_gyro.add_variable("stabilizer.roll", "float")
        self.lg_stab_gyro.add_variable("stabilizer.pitch", "float")
        self.lg_stab_gyro.add_variable("stabilizer.yaw", "float")
        self.lg_stab_gyro.add_variable("gyro.x", "float")
        self.lg_stab_gyro.add_variable("gyro.y", "float")
        self.lg_stab_gyro.add_variable("gyro.z", "float")

        scf.cf.log.add_config(self._lg_esti)
        scf.cf.log.add_config(self.lg_stab_gyro)

        # This callback will receive the data
        self._lg_esti.data_received_cb.add_callback(self._log_data)

        # Start the logging
        self._lg_esti.start()

        # This callback will receive the data
        self.lg_stab_gyro.data_received_cb.add_callback(self._log_data)

        # Start the logging
        self.lg_stab_gyro.start()

    def _log_error(self, logconf, msg):
        """Callback from the log API when an error occurs."""
        print(f"Error when logging {logconf.name}: {msg}")

    def _log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives."""
        # print(f'[{timestamp}][{logconf.name}]: ', end='')

        if "gyro" in logconf.name:
            info = [value for name, value in data.items()]
            self.rpy = info[0:3]
            self.gyro = info[3:6]
            # for name, value in data.items():
            #     print(f'{name}: {value:3.3f} ', end='')
        else:
            info = [value for name, value in data.items()]
            self.pos = info[0:3]
            self.vel = info[3:6]
            # for name, value in data.items():
            #     print(f'{name}: {value:3.3f} ', end='')

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie at the specified address)."""
        print(f"Connection to {link_uri} failed: {msg}")
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e Crazyflie moves out of range)."""
        print(f"Connection to {link_uri} lost: {msg}")

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)."""
        print("Disconnected from %s" % link_uri)
        self.is_connected = False

    def stop_logs(self):
        """Stop logging."""
        self._cf.close_link()
