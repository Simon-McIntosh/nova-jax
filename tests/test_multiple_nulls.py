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
    x, z = np.linspace(-1, 1, 13), np.linspace(0, 5, 9)
    x2d, z2d = np.meshgrid(x, z, indexing="ij")
    data = xarray.Dataset(dict(zip("xz", (x, z))))
    data["x2d"] = ("x", "z"), x2d
    data["z2d"] = ("x", "z"), z2d
    data["psi"] = ("x", "z"), fun(x2d, z2d)
    return data


@pytest.mark.parametrize(("xo", "zo"), itertools.product([-0.7, 0, 0.2], [0.4, 4.4, 3]))
def test_multi_null(xo, zo):
    data = generate(lambda x, z: (x - xo) ** 2 + (z - zo) ** 2 + 7.3)
    fieldnull = FieldNull(data, maxsize=1)
    point_psi = fieldnull.update_null(data.psi.data.ravel())[0][0, :3]
    assert np.allclose(point_psi, (xo, zo, 7.3), atol=1e-5)


def test_x_point(plot=False):
    data = generate(lambda x, z: np.exp(-(x**2)) * np.sin(x) + np.cos(2.5 * z))
    fieldnull = FieldNull(data, maxsize=5)
    null = fieldnull.update_null(data.psi.data.ravel())

    if plot:
        plt.contour(data.x, data.z, data.psi.T)
        plt.plot(*null[0][:, :2].T, "C3o")
        plt.plot(*null[1][:, :2].T, "C3x")

    null = fieldnull.update_null(data.psi.data.ravel())
    print(null)


test_x_point(True)
assert False

if __name__ == "__main__":
    pytest.main([__file__])
