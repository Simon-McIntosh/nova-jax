#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:01:45 2024

@author: mcintos
"""

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import xarray


def categorize_1d(data, stencil):
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
    for index in stencil:
        center = data[index[0]]
        sign = data[index[-1]] > center
        count = 0
        for k in range(1, 7):
            _sign = data[index[k]] > center
            if _sign != sign:
                count += 1
                sign = _sign
        if count == 0:
            o_mask = o_mask.at[index[0]].set(True)
        if count == 4:
            x_mask = x_mask.at[index[0]].set(True)
    return o_mask, x_mask


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

plt.triplot(
    plasmagrid.x,
    plasmagrid.z,
    plasmagrid.triangles,
    lw=1.5,
    color="C0",
    alpha=0.2,
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

plt.plot(plasmagrid.x[o_mask], plasmagrid.z[o_mask], "r.")


plt.axis("equal")
plt.axis("off")
