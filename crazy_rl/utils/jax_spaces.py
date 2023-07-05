"""Taken from Gymnax."""
from typing import Tuple
from typing_extensions import override

import chex
import jax
import jax.numpy as jnp


class Space:
    """Minimal jittable class for abstract gymnax space."""

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample a random element from this space.

        Args:
            rng: key for random number generator
        """
        raise NotImplementedError

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete gymnax spaces.

    TODO: For now this is a 1d space. Make composable for multi-discrete.
    """

    def __init__(self, num_categories: int):
        """Create a discrete space with a given number of categories."""
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int_

    @override
    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(rng, shape=self.shape, minval=0, maxval=self.n).astype(self.dtype)

    @override
    def contains(self, x: jnp.int_) -> bool:
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(Space):
    """Minimal jittable class for array-shaped gymnax spaces.

    TODO: Add unboundedness - sampling from other distributions, etc.
    """

    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        """Create a box space with a given shape and bounds."""
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    @override
    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)

    @override
    def contains(self, x: jnp.int_) -> bool:
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond
