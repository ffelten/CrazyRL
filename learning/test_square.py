import cflib
from cflib.crazyflie.swarm import CachedCfFactory, Swarm

from crazy_rl.utils.utils import LoggingCrazyflie, run_land, run_square, run_take_off


if __name__ == "__main__":
    # Init swarm config of crazyflie
    cflib.crtp.init_drivers()
    uris = {
        "radio://0/4/2M/E7E7E7E700",
        # "radio://0/4/2M/E7E7E7E701",
        # Add more URIs if you want more copters in the swarm
    }
    # uri = 'radio://0/4/2M/E7E7E7E7' + str(id).zfill(2) # you can browse the drone_id and add as this code at the end of the uri

    # the Swarm class will automatically launch the method in parameter of parallel_safe method
    factory = CachedCfFactory(rw_cache="./cache")
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(LoggingCrazyflie)
        swarm.parallel_safe(run_take_off)
        swarm.parallel_safe(run_square)
        swarm.parallel_safe(run_land)
