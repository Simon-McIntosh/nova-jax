#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:01:45 2024

@author: mcintos
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray

from nova_jax import select


@jax.jit
def _categorize_1d(data, stencil):
    """Categorize points in 1d hexagonal grid.

    Count number of sign changes whilst traversing neighbour point loop.

        - 0: minima / maxima point
        - 2: regular point
        - 4: saddle point

    From On detecting all saddle points in 2D images, A. Kuijper

    """
    npoint = len(data)
    o_mask = jnp.full(npoint, False)
    x_mask = jnp.full(npoint, False)

    def stencil_index(i, val):
        o_mask, x_mask = val
        index = stencil[i]
        center = data[index[0]]

        def stencil_vertex(k, val):
            sign, count = val
            _sign = data[index[k]] > center
            sign_change = _sign != sign
            count = jnp.where(sign_change, count + 1, count)
            sign = jnp.where(sign_change, _sign, sign)
            return [sign, count]

        sign = data[index[-1]] > center
        count = jax.lax.fori_loop(1, 7, stencil_vertex, [sign, 0])[1]
        o_mask = o_mask.at[index[0]].set(count == 0)
        x_mask = x_mask.at[index[0]].set(count == 4)
        return [o_mask, x_mask]

    return jax.lax.fori_loop(0, len(stencil), stencil_index, [o_mask, x_mask])

    """
    for index in stencil:
        center = data[index[0]]
        sign = data[index[-1]] > center
        count = 0
        for k in range(1, 7):
            _sign = data[index[k]] > center
            sign_change = _sign != sign
            count = jnp.where(sign_change, count + 1, count)
            sign = jnp.where(sign_change, _sign, sign)
            # if _sign != sign:
            #    count += 1
            #    sign = _sign
        # jnp.where(count == 0)
        # if count == 0:
        o_mask = o_mask.at[index[0]].set(count == 0)
        # if count == 4:
        x_mask = x_mask.at[index[0]].set(count == 4)
    return o_mask, x_mask
    """


def _index_1d(x_coordinate, z_coordinate, mask):
    index = jnp.where(mask)[0]
    point_number = len(index)
    points = jnp.zeros((point_number, 2), dtype=float)
    for i in range(point_number):  # pylint: disable=not-an-iterable
        points = points.at[i, 0].set(x_coordinate[index[i]])
        points = points.at[i, 1].set(z_coordinate[index[i]])
    return index, points


def _unique(nulls, decimals=3):
    """Return unique field nulls."""
    points = jnp.array([null[0] for null in nulls])
    psi = jnp.array([null[1] for null in nulls])
    null_type = np.array([null[2] for null in nulls])
    points, index = np.unique(points.round(decimals), axis=0, return_index=True)
    return {"points": points, "psi": psi[index], "null_type": null_type[index]}


def _subnull_1d(x_coordinate, z_coordinate, stencil, stencil_index, index, psi):
    """Return unique field nulls from 1d unstructured grid."""
    nulls = []
    for i in index:
        stencil_vertex = stencil[select.bisect(stencil_index, i)]
        x_cluster = x_coordinate[stencil_vertex]
        z_cluster = z_coordinate[stencil_vertex]
        jax.debug.print("x_cluster {x_cluster}", x_cluster=x_cluster)
        jax.debug.print("z_cluster {z_cluster}", z_cluster=z_cluster)

        # psi_cluster = psi[stencil_vertex]
        # nulls.append(select.subnull(x_cluster, z_cluster, psi_cluster))
    # return {"index": index} | _unique(nulls)


plasmagrid = xarray.open_dataset("plasmagrid.nc")
levelset = data = xarray.open_dataset("levelset.nc")
data = xarray.open_dataset("data.nc")


itime = 20

current = data.current[itime]
passive_current = data.passive_current[itime]
plasma_current = data.ip[itime]

currents = np.r_[current, passive_current, plasma_current]


def psi(currents):
    """Return levelset flux vector."""
    return jnp.matmul(levelset.Psi.data, currents)


def psi_plasma(currents):
    return jnp.matmul(plasmagrid.Psi.data, currents)


dpsi_dI = jax.jacfwd(psi)(currents)

assert np.allclose(levelset.Psi, dpsi_dI)

# dpsi_dI = jax.grad(psi)
o_mask, x_mask = categorize_1d(psi_plasma(currents), plasmagrid.stencil.data)
index, points = _index_1d(plasmagrid.x, plasmagrid.z, o_mask)

_subnull_1d(
    plasmagrid.x.data,
    plasmagrid.z.data,
    plasmagrid.stencil.data,
    plasmagrid.stencil_index.data,
    index,
    psi_plasma(currents),
)


plt.figure(figsize=(5, 9))
contour = plt.contour(
    levelset.x,
    levelset.z,
    psi(currents).reshape(levelset.sizes["x"], levelset.sizes["z"]).T,
    levels=71,
    colors="gray",
    linestyles="solid",
    linewidths=1.5,
)

plt.tricontour(
    plasmagrid.x,
    plasmagrid.z,
    plasmagrid.triangles,
    psi_plasma(currents),
    levels=contour.levels,
    colors="C0",
    linestyles="solid",
    linewidths=1.5,
)

plt.triplot(
    plasmagrid.x,
    plasmagrid.z,
    plasmagrid.triangles,
    lw=1.5,
    color="C0",
    alpha=0.2,
)

plt.plot(plasmagrid.x[o_mask], plasmagrid.z[o_mask], "r.")

plt.axis("equal")
plt.axis("off")
