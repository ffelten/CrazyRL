env = Catch(
        drone_ids=np.arange(8),
        render_mode="human",
        init_flying_pos=np.array(
            [
                [-0.7, -0.5, 1.5],
                [-0.8, 0.5, 0.5],
                [1.0, 0.5, 1.5],
                [0.5, 0.0, 0.5],
                [0.5, -0.5, 1.0],
                [2.0, 2.5, 2.0],
                [2.0, 1.0, 2.5],
                [0.5, 0.5, 0.5],
            ]
        ),
        init_target_location=np.array([0.0, 0.5, 1.5]),
        target_speed=0.15,
        size=5,
    )