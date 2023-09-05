env = Escort(
            num_drones=num_drones,
            init_flying_pos=jnp.array(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.5, 1.5],
                    [2.0, 0.5, 1.0],
                    # [2.0, 2.5, 2.0],
                    # [2.0, 1.0, 2.5],
                    # [0.5, 0.5, 0.5],
                ]
            ),
            init_target_location=jnp.array([-1.0, -1.5, 2.0]),
            final_target_location=jnp.array([1.0, 1.5, 1.0])
        )