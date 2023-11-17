env: ParallelEnv = Circle(
            drone_ids=drones_ids,
            render_mode="real",
            init_flying_pos=np.array([[-0.5, 0.0, 1.5], [0.0, 0.5, 0.5], [0.5, 0.0, 1.5]]),
            swarm=swarm,
        )