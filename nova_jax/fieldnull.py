"""Methods for calculating the position and value of x-points and o-points."""

from dataclasses import dataclass, field

from functools import cached_property, partial
import jax
import jax.numpy as jnp
import numpy as np
import xarray

from nova_jax.array import Array
from nova_jax.categorize import Null
from nova_jax import select

"""
from nova import njit
from nova.biot.array import Array
from nova.graphics.plot import Plot
from nova.geometry import select
from nova.geometry.pointloop import PointLoop
"""


@dataclass
class DataNull(Array):
    """Store sort and remove field nulls."""

    subgrid: bool = True
    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    loop: np.ndarray | None = field(repr=False, default=None)
    array_attrs: list[str] = field(
        default_factory=lambda: ["x", "z", "stencil", "stencil_index"]
    )
    data_o: dict[str, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    data_x: dict[str, np.ndarray] = field(init=False, default_factory=dict, repr=False)

    def update_masks(self, mask_o, mask_x, psi):
        """Update null points."""
        for null, mask in zip("ox", [mask_o, mask_x]):
            setattr(self, f"data_{null}", self.update_mask(mask, psi))

    def update_mask(self, mask, psi):
        """Return masked point data dict."""
        if psi.ndim == 1:
            return self.update_mask_1d(mask, psi)
        return self.update_mask_2d(mask, psi)

    @staticmethod
    def _empty_mask():
        """Return empty dict structure when all(mask==0)."""
        index, points = np.empty((0, 2), int), np.empty((0, 2), float)
        return dict(index=index, points=points, psi=np.empty(0))

    def _select(self, points, index):
        """Select subset of points within loop when loop is not None."""
        if self.loop is None:
            return points, index
        subindex = PointLoop(points).update(self.loop)
        return points[subindex], index[subindex]

    def update_mask_1d(self, mask, psi):
        """Return masked data dict from 1D input."""
        try:
            index, points = self._index_1d(self["x"], self["z"], mask)
        except IndexError:  # catch empty mask
            return self._empty_mask()
        points, index = self._select(points, index)
        if self.subgrid:
            return {"index": index} | self._subnull_1d(index, psi)
        return {"index": index, "points": points, "psi": psi[index]}

    def update_mask_2d(self, mask, psi):
        """Return masked data dict from 2D input."""
        try:
            index, points = self._index_2d(self["x"], self["z"], mask)
        except IndexError:
            return self._empty_mask()
        points, index = self._select(points, index)
        if self.subgrid:
            return {"index": index} | self._subnull_2d(index, psi)
        return {
            "index": index,
            "points": points,
            "psi": np.array([psi[tuple(i)] for i in index]),
        }

    @staticmethod
    def _unique(nulls, decimals=3):
        """Return unique field nulls."""
        points = np.array([null[0] for null in nulls])
        psi = np.array([null[1] for null in nulls])
        null_type = np.array([null[2] for null in nulls])
        _, index = np.unique(points.round(decimals), axis=0, return_index=True)
        return {
            "points": points[index],
            "psi": psi[index],
            "null_type": null_type[index],
        }

    def _subnull_1d(self, index, psi):
        """Return unique field nulls from 1d unstructured grid."""
        stencil_index = self["stencil_index"]
        nulls = []
        for i in index:
            stencil_vertex = self["stencil"][select.bisect(stencil_index, i)]
            x_cluster = self["x"][stencil_vertex]
            z_cluster = self["z"][stencil_vertex]
            psi_cluster = psi[stencil_vertex]
            nulls.append(select.subnull(x_cluster, z_cluster, psi_cluster))
        return {"index": index} | self._unique(nulls)

    def _subnull_2d(self, index, psi2d):
        """Return unique field nulls from 2d grid."""
        x_coordinate, z_coordinate = self["x"], self["z"]
        nulls = []
        for i, j in index:
            x2d, z2d = np.meshgrid(
                x_coordinate[i - 1 : i + 2], z_coordinate[j - 1 : j + 2], indexing="ij"
            )
            x_cluster, z_cluster = x2d.flatten(), z2d.flatten()
            psi_cluster = psi2d[i - 1 : i + 2, j - 1 : j + 2].flatten()
            nulls.append(select.subnull(x_cluster, z_cluster, psi_cluster))
        return dict(index=index) | self._unique(nulls)

    @staticmethod
    def _index_1d(x_coordinate, z_coordinate, mask):
        index = np.where(mask)[0]
        point_number = len(index)
        points = np.empty((point_number, 2), dtype=np.float64)
        for i in range(point_number):
            points[i, 0] = x_coordinate[index[i]]
            points[i, 1] = z_coordinate[index[i]]
        return index, points

    @staticmethod
    def _index_2d(x_coordinate, z_coordinate, mask):
        index = np.asarray(list(zip(*np.where(mask))))
        point_number = len(index)
        points = np.empty((point_number, 2), dtype=np.float64)
        for i in range(point_number):
            points[i, 0] = x_coordinate[index[i][0]]
            points[i, 1] = z_coordinate[index[i][1]]
        return index, points

    def delete(self, null: str, index):
        """Delete elements in data specified by index.

        Parameters
        ----------
            index: slice, int or array of ints
                index to remove.

        """
        data = getattr(self, f"data_{null}")
        for attr in data:
            data[attr] = np.delete(data[attr], index, axis=0)

    def plot(self, axes=None):
        """Plot null points."""
        self.get_axes(axes)
        if self.o_point_number > 0:
            self.axes.plot(
                *self.data_o["points"].T, "o", ms=4, mec="C3", mew=1, mfc="none"
            )
        if self.x_point_number > 0:
            self.axes.plot(
                *self.data_x["points"].T, "x", ms=6, mec="C3", mew=1, mfc="none"
            )


@dataclass
class FieldNull(DataNull):
    """Calculate positions of all field nulls."""

    @property
    def o_points(self):
        """Return o-point locations."""
        return self.data_o["points"]

    @property
    def o_psi(self):
        """Return flux values at o-point locations."""
        return self.data_o["psi"]

    @property
    def o_point_number(self):
        """Return o-point number."""
        return len(self.o_psi)

    @property
    def x_points(self):
        """Return x-point locations."""
        return self.data_x["points"]

    @property
    def x_psi(self):
        """Return flux values at x-point locations."""
        return self.data_x["psi"]

    @property
    def x_point_number(self):
        """Return x-point number."""
        return len(self.x_psi)

    @cached_property
    def stencil(self):
        """Return grid stencil."""
        if "stencil" in self.data:
            return self.data["stencil"].data
        patch = np.array([(0, 0), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)])
        return np.ravel_multi_index(
            np.indices((self.data.sizes["x"] - 2, self.data.sizes["z"] - 2)).reshape(
                2, -1, 1
            )
            + 1
            + patch.T[:, np.newaxis],
            (self.data.sizes["x"], self.data.sizes["z"]),
        )

    @cached_property
    def coordinate_stencil(self):
        """Return stencil geometry."""
        if "stencil" in self.data:  # unstructured grid
            return np.c_[self.data.x, self.data.z][self.stencil]
        return np.c_[
            self.data.x2d.data.ravel(),
            self.data.z2d.data.ravel(),
        ][self.stencil]

    # def update_mask(self):
    #    return {"index": index, "points": points, "psi": psi[index]}


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plasmagrid = xarray.open_dataset("plasmagrid.nc")
    levelset = xarray.open_dataset("levelset.nc")
    data = xarray.open_dataset("data.nc")

    def psi_plasma(currents):
        """Return plasmagrid flux map."""
        return np.matmul(plasmagrid.Psi.data, currents)

    def psi_levelset(currents):
        """Return levelset flux map."""
        return np.matmul(levelset.Psi.data, currents)

    itime = 20

    current = data.current[itime]
    passive_current = data.passive_current[itime]
    plasma_current = data.ip[itime]
    currents = np.r_[current, passive_current, plasma_current]

    psi_1d = psi_plasma(currents)
    psi_2d = psi_levelset(currents)

    fieldnull = FieldNull(data=levelset)

    null = Null(fieldnull.stencil, fieldnull.coordinate_stencil, 2)

    print(null.update(psi_2d))
    """

    def o_point_r(psi_2d):
        fieldnull.update_null(psi_2d)
        return fieldnull.o_points[0][0]
    """

    plt.figure(figsize=(5, 9))

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
        levels=71,
        colors="C0",
        linestyles="solid",
        linewidths=1.5,
    )
    plt.axis("equal")
    plt.axis("off")

    """


    (o_point_number, x_point_number), count = categorize_1d(psi_1d, stencil)

    plt.plot(plasmagrid.x[o_mask], plasmagrid.z[o_mask], "r.")

    o_mask, x_mask = categorize_2d(psi_2d)
    plt.plot(levelset.x2d.data[o_mask], levelset.z2d.data[o_mask], "r.")
    plt.plot(levelset.x2d.data[x_mask], levelset.z2d.data[x_mask], "rx")

    plt.tricontour(
        plasmagrid.x,
        plasmagrid.z,
        plasmagrid.triangles,
        psi_plasma(currents),
        levels=71,
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


    """
