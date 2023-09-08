env = Catch(
            drone_ids=drones_ids,
            render_mode="real",
            init_flying_pos=jnp.array(
                [
                    # [-0.7, -0.5, 1.5],
                    [-0.8, 0.5, 0.5],
                    [1.0, 0.5, 1.5],
                    [0.5, 0.0, 0.5],
                    [0.5, -0.5, 1.0],
                    # [2.0, 2.5, 2.0],
                    # [2.0, 1.0, 2.5],
                    # [0.5, 0.5, 0.5],
                ]
            ),
            init_target_location=jnp.array([0.0, 0.5, 1.5]),
            target_speed=0.1,
            size=1.3,
            target_id=target_id,
            swarm=swarm,
        )