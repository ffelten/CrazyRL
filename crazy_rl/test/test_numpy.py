"""Tests for the existing Crazyflie environments."""
import numpy as np
from pettingzoo.test.parallel_test import parallel_api_test

from crazy_rl.multi_agent.numpy.catch.catch import Catch
from crazy_rl.multi_agent.numpy.circle.circle import Circle
from crazy_rl.multi_agent.numpy.escort.escort import Escort
from crazy_rl.multi_agent.numpy.hover.hover import Hover
from crazy_rl.multi_agent.numpy.surround.surround import Surround


def test_hover():
    """Test for the hover environment."""
    parallel_api_test(
        Hover(
            drone_ids=np.array([0, 1]),
            render_mode=None,
            init_flying_pos=np.array([[0, 0, 1], [1, 1, 1]]),
        ),
        num_cycles=10,
    )


def test_circle():
    """Test for the circle environment."""
    parallel_api_test(
        Circle(
            drone_ids=np.array([0, 1]),
            render_mode=None,
            init_flying_pos=np.array([[0, 0, 1], [1, 1, 1]]),
        ),
        num_cycles=10,
    )


def test_surround():
    """Test for the surround environment."""
    parallel_api_test(
        Surround(
            drone_ids=np.array([0, 1, 2, 3, 4]),
            render_mode=None,
            init_flying_pos=np.array([[0, 0, 1], [2, 1, 1], [0, 1, 1], [2, 2, 1], [1, 0, 1]]),
            target_location=np.array([1, 1, 2.5]),
        ),
        num_cycles=10,
    )


def test_escort():
    """Test for the escort environment."""
    parallel_api_test(
        Escort(
            drone_ids=np.array([0, 1, 2, 3]),
            render_mode=None,
            init_flying_pos=np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [2, 2, 1]]),
            init_target_location=np.array([1, 1, 2.5]),
            final_target_location=np.array([-2, -2, 3]),
            num_intermediate_points=150,
        ),
        num_cycles=10,
    )


def test_catch():
    """Test for the catch environment."""
    parallel_api_test(
        Catch(
            drone_ids=np.array([0, 1, 2, 3]),
            render_mode=None,
            init_flying_pos=np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [2, 2, 1]]),
            init_target_location=np.array([1, 1, 2.5]),
            target_speed=0.1,
        ),
        num_cycles=10,
    )
