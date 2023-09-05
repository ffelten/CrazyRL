# author: Daniel H. Stolfi
# project: ADARS
# Geometry updater
"""Save the room's geometry into the crazyflie."""

import logging
import sys
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import LighthouseMemHelper
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.localization import lighthouse_config_manager
from cflib.utils import uri_helper


logging.basicConfig(level=logging.ERROR)


class WriteMem:
    """Write into the remote memory."""

    def __init__(self, uri, geo_dict, calib_dict):
        self._event = Event()

        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            helper = LighthouseMemHelper(scf.cf)

            helper.write_geos(geo_dict, self._data_written)
            self._event.wait()

            self._event.clear()

            helper.write_calibs(calib_dict, self._data_written)
            self._event.wait()

    def _data_written(self, success):
        if success:
            print("Data written")
        else:
            print("Write failed")

        self._event.set()


class ReadMem:
    """Read from the remote memory."""

    def __init__(self, uri, verbose=False):
        self._event = Event()
        self.verbose = verbose
        self.geo_data = {}
        self.calib_data = {}

        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            print(
                "Firmware: {} {} {}".format(
                    scf.cf.param.get_value("firmware.revision0"),
                    scf.cf.param.get_value("firmware.revision1"),
                    scf.cf.param.get_value("firmware.modified"),
                )
            )
            # print(scf.cf.toc.get_element_by_complete_name('pm.vbatMV'))

            helper = LighthouseMemHelper(scf.cf)

            helper.read_all_geos(self._geo_read_ready)
            self._event.wait()

            self._event.clear()

            helper.read_all_calibs(self._calib_read_ready)
            self._event.wait()

    def _geo_read_ready(self, geo_data):
        self.geo_data = geo_data
        if self.verbose:
            for id, data in geo_data.items():
                print("---- Geometry for base station", id + 1)
                data.dump()
                print()
        self._event.set()

    def _calib_read_ready(self, calib_data):
        self.calib_data = calib_data
        if self.verbose:
            for id, data in calib_data.items():
                print("---- Calibration data for base station", id + 1)
                data.dump()
                print()
        self._event.set()


def save_and_check(filename, drone_id, verbose):
    """Save the configuration and check that it was successful."""
    # URI to the Crazyflie to connect to
    drone_id = "0" + drone_id.upper()
    drone_id = drone_id[-2:]
    uri = uri_helper.uri_from_env(default=f"radio://0/4/2M/E7E7E7E7{drone_id}")

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    print(f"Uploading {filename} to {uri}")

    config = lighthouse_config_manager.LighthouseConfigFileManager().read(filename)

    WriteMem(uri, config[0], config[1])

    print(f"Reading from {uri}")

    memory = ReadMem(uri, verbose)

    same = True
    for lh in [0, 1]:
        same = same and config[0][lh].origin == memory.geo_data[lh].origin
        same = same and config[0][lh].rotation_matrix == memory.geo_data[lh].rotation_matrix
        same = same and config[0][lh].valid == memory.geo_data[lh].valid
        for sw in [0, 1]:
            same = same and config[1][lh].uid == memory.calib_data[lh].uid
            same = same and config[1][lh].valid == memory.calib_data[lh].valid
            same = same and config[1][lh].sweeps[sw].phase == memory.calib_data[lh].sweeps[sw].phase
            same = same and config[1][lh].sweeps[sw].tilt == memory.calib_data[lh].sweeps[sw].tilt
            same = same and config[1][lh].sweeps[sw].curve == memory.calib_data[lh].sweeps[sw].curve
            same = same and config[1][lh].sweeps[sw].gibmag == memory.calib_data[lh].sweeps[sw].gibmag
            same = same and config[1][lh].sweeps[sw].gibphase == memory.calib_data[lh].sweeps[sw].gibphase
            same = same and config[1][lh].sweeps[sw].ogeemag == memory.calib_data[lh].sweeps[sw].ogeemag
            same = same and config[1][lh].sweeps[sw].ogeephase == memory.calib_data[lh].sweeps[sw].ogeephase

    return same


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} file.yaml drone_id[s] [verbose]")
        exit(-1)

    filename = sys.argv[1]
    drone_ids = sys.argv[2].split(",")
    verbose = False
    if len(sys.argv) == 4:
        verbose = sys.argv[3] == "1"

    for drone_id in drone_ids:
        if save_and_check(filename, drone_id, verbose):
            print("SUCCESS!")
        else:
            print("FAIL")
