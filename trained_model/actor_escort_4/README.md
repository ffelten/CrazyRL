env = Escort(
            drone_ids=drones_ids,
            render_mode="real",
            init_flying_pos=np.array(
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
            # target_location=jnp.array([0.0, 0.5, 1.5]),
            init_target_location=np.array([-0.1, 0.6, 1.1]),
            final_target_location=np.array([1.2, -1.3, 2.3]),
            target_id=target_id,
            swarm=swarm,
        )