```python
Surround(
    drone_ids=drones_ids,
    render_mode="real",
    init_flying_pos=np.array(
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
    target_location=np.array([0.0, 0.5, 1.5]),
    target_id=target_id,
)

```
