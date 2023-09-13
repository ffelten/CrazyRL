For the Pareto front generation

env = Surround(
            num_drones=num_drones,
            init_flying_pos=jnp.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 2.0, 2.0],
                    [2.0, 0.5, 1.0],
                    [2.0, 2.5, 2.0],
                    [2.0, 1.0, 2.5],
                    [0.5, 0.5, 0.5],
                ]
            ),
            target_location=jnp.array([1.0, 1.0, 2.0]),
            multi_obj=True,
            size=5,
            # target_speed=0.15,
            # final_target_location=jnp.array([-2.0, -2.0, 1.0]),
        )


For measuring the time with different number of policies
num_drones = 2
        env = Surround(
            num_drones=num_drones,
            init_flying_pos=jnp.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    # [1.0, 0.0, 1.0],
                    # [1.0, 2.0, 2.0],
                    # [2.0, 0.5, 1.0],
                    # [2.0, 2.5, 2.0],
                    # [2.0, 1.0, 2.5],
                    # [0.5, 0.5, 0.5],
                ]
            ),
            target_location=jnp.array([1.0, 1.0, 2.0]),
            multi_obj=True,
            size=5,
            # target_speed=0.15,
            # final_target_location=jnp.array([-2.0, -2.0, 1.0]),
        )