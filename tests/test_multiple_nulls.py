# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:57:07 2024

@author: mcintos
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray

from nova_jax.fieldnull import FieldNull


def generate(fun):
    x, z = np.linspace(-1.5, 1.5, 20), np.linspace(0, 3, 20)
    x2d, z2d = np.meshgrid(x, z, indexing="ij")
    data = xarray.Dataset(dict(zip("xz", (x, z))))
    data["x2d"] = ("x", "z"), x2d
    data["z2d"] = ("x", "z"), z2d
    data["psi"] = ("x", "z"), fun(x2d, z2d)
    return data


def plot_null(data, null):
    plt.contour(data.x, data.z, data.psi.T)
    plt.plot(*null[0][:, :2].T, "C3o")
    plt.plot(*null[1][:, :2].T, "C3x")


@pytest.mark.parametrize(
    ("xo", "zo"), itertools.product([-0.7, -0.3, 0.2], [0.4, 2.4, 1.2])
)
def test_multi_null(xo, zo, plot=False):
    data = generate(lambda x, z: (x - xo) ** 2 + (z - zo) ** 2 + 7.3)
    fieldnull = FieldNull(data, maxsize=1)
    null = fieldnull.update_null(data.psi.data.ravel())

    if plot:
        plot_null(data, null)
    point_psi = null[0][0, :3]
    assert np.allclose(point_psi, (xo, zo, 7.3), atol=1e-4)


def test_x_point(plot=False):
    data = generate(lambda x, z: np.sin(2 * x) * np.cos(3 * z))
    fieldnull = FieldNull(data, maxsize=5)
    null = fieldnull.update_null(data.psi.data.ravel())

    if plot:
        plot_null(data, null)

    null = fieldnull.update_null(data.psi.data.ravel())

    assert np.isclose(np.nanmin(null[0][:, 0]), -np.pi / 4, atol=1e-3)
    assert np.isclose(np.nanmax(null[0][:, 0]), np.pi / 4, atol=1e-3)
    assert np.isclose(np.nanmin(null[0][:, 1]), np.pi / 3, atol=1e-3)
    assert np.isclose(np.nanmax(null[0][:, 1]), 2 * np.pi / 3, atol=1e-3)
    assert np.allclose(null[1][:3, 0], 0, atol=1e-2)
    assert np.isclose(np.nanmin(null[1][:, 1]), np.pi / 6, atol=1e-3)
    assert np.isclose(np.nanmax(null[1][:, 1]), 5 * np.pi / 6, atol=8e-2)


if __name__ == "__main__":
    pytest.main([__file__])
