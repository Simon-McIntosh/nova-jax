"""Group fieldnull categorization algorithms."""

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

from nova_jax import select


@dataclass
@jax.tree_util.register_pytree_node_class
class Null:
    """Locate and label field nulls on structured and unstructured grids."""

    stencil: jnp.ndarray
    coordinate_stencil: jnp.ndarray
    maxsize: int = 20

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (self.stencil, self.coordinate_stencil)
        aux_data = {"maxsize": self.maxsize}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Return unflattened pytree."""
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def zero_cross_count(number, patch_array):
        """Count the total number of sign changes when traversing vertex patch.

            - 0: minima / maxima point
            - 2: regular point
            - 4: saddle point

        From On detecting all saddle points in 2D images, A. Kuijper
        """

        def zero_cross(carry, value):
            """Increment zero crossing counter and update state."""
            count, sign, center = carry
            _sign = value > center
            sign_change = _sign != sign
            count += sign_change
            sign = jnp.where(sign_change, _sign, sign)
            return (count, sign, center), None

        center = patch_array[0]
        sign = patch_array[-1] > center
        count = jax.lax.scan(zero_cross, (0, sign, center), patch_array[1:])[0][0]
        number += count == 0
        number += count == 4
        return number, count

    @jax.jit
    def categorize(self, data):
        """Categorize points in 1d hexagonal grid."""
        number, count = jax.lax.scan(self.zero_cross_count, 0, data)
        index = jnp.where((count == 0) | (count == 4), size=self.maxsize)[0]
        return number, index

    @jax.jit
    def interpolate(self, number, cluster):
        """Interpolate subnull."""

        def subnull(carry, cluster):
            carry += 1
            return carry, jnp.where(
                carry <= number, select.subnull(cluster.T), jnp.array([0, 0, 0, 0])
            )

        return jax.lax.scan(subnull, 0, cluster)[1]

    @jax.jit
    def update(self, psi):
        """Return subgrid interpolated field nulls."""
        psi_stencil = psi[self.stencil]
        number, index = self.categorize(psi_stencil)
        cluster = jnp.append(
            self.coordinate_stencil[index],
            psi_stencil[index, :, jnp.newaxis],
            axis=-1,
        )
        return self.interpolate(number, cluster)
