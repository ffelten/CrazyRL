env: ParallelEnv = Circle(
        drone_ids=np.arange(num_drones),
        init_flying_pos=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
    )
