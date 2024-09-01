import numpy as np

from numpy import linspace, zeros

from .boundary import Boundary
from .constants import btype
from .cell import Cell
from .error import SFXM


class Domain:
    """
    A class representing a domain containing cells and boundaries.

    Parameters
    ----------
    cells : list of Cell
        The cells in the domain.
    left_boundaries : list of Boundary
        The left boundaries of the domain.
    right_boundaries : list of Boundary
        The right boundaries of the domain.
    components : list of str
        The names of the components in the domain.
    """

    def __init__(
        self, cells: list[Cell], left_boundaries: list[Boundary], right_boundaries: list[Boundary], components: list[str]
    ):
        """
        Initialize a `Domain` object.
        """
        # Boundaries list contains both left and right
        # nb indicates number on each side
        self._nb = {btype.LEFT: int(
            len(left_boundaries)), btype.RIGHT: int(len(right_boundaries))}

        self._nx = len(cells)
        # Domain is a list of left boundary, interior cells and right boundaries
        self._domain = [*left_boundaries, *
                        cells, *right_boundaries]
        self._components = components

    @ classmethod
    def from_size(
        cls,
        nx: int,
        nb_left: int,
        nb_right: int,
        components: list[str],
        xmin: float = 0.0,
        xmax: float = 1.0,
    ):
        """
        Initialize a `Domain` object with a uniform grid of cells.

        Parameters
        ----------
        nx : int
            The number of cells in the domain.
        nb_left: int
            The number of ghost cells on the left side of the domain.
        nb_right: int
            The number of ghost cells on the right side of the domain.
        components : list of str
            The names of the components in the domain.
        xmin : float, optional
            The minimum x-value of the domain. Default is 0.0.
        xmax : float, optional
            The maximum x-value of the domain. Default is 1.0.

        Returns
        -------
        Domain
            The initialized `Domain` object.
        """

        # Initialize a uniform grid
        xarr = linspace(xmin, xmax, nx)
        nv = len(components)
        interior = [Cell(x, zeros(nv)) for x in xarr]

        # Create boundaries
        dx = (xmax - xmin) / nx
        left_boundaries = [
            Boundary(xmin - (i + 1) * dx, btype.LEFT, zeros(nv))
            for i in range(nb_left)
        ]
        right_boundaries = [
            Boundary(xmax + (i + 1) * dx, btype.RIGHT, zeros(nv))
            for i in range(nb_right)
        ]

        return Domain(interior, left_boundaries, right_boundaries, components)

    def ilo(self):
        """
        Get the index of the leftmost interior cell.

        Returns
        -------
        int
            The index of the leftmost interior cell.
        """
        return self._nb[btype.LEFT]

    def ihi(self):
        """
        Get the index of the rightmost interior cell.

        Returns
        -------
        int
            The index of the rightmost interior cell.
        """
        return self._nb[btype.LEFT] + self._nx - 1

    def nb(self, dir: btype):
        """
        Get the number of ghost cells on each side of the domain.

        Returns
        -------
        int
            The number of ghost cells on each side of the domain.
        """
        return self._nb[dir]

    def cells(self, interior=False):
        """
        Get the cells in the domain.

        Returns
        -------
        list of Cell
            The cells in the domain.
        """
        return self._domain if not interior else self.interior()

    def boundaries(self):
        """
        Get the boundaries of the domain.

        Returns
        -------
        tuple of Boundary
            The left and right boundaries of the domain.
        """
        nb_left, nb_right = self._nb[btype.LEFT], self._nb[btype.RIGHT]
        return self._domain[: nb_left], self._domain[-nb_right:]

    def interior(self):
        """
        Get the interior cells in the domain.

        Returns
        -------
        list of Cell
            The interior cells in the domain.
        """
        nb_left, nb_right = self._nb[btype.LEFT], self._nb[btype.RIGHT]
        return self._domain[nb_left: -nb_right]

    def set_interior(self, cells):
        """
        Set the interior cells in the domain.

        Parameters
        ----------
        cells : list of Cell
            The new interior cells in the domain.
        """
        self._nx = len(cells)
        nb_left, nb_right = self._nb[btype.LEFT], self._nb[btype.RIGHT]
        self._domain = [*self._domain[: nb_left],
                        *cells, *self._domain[-nb_right:]]

    def num_components(self):
        """
        Get the number of components/variables associated in the domain.

        Returns
        -------
        int
            The number of components in the domain.
        """
        return len(self._components)

    def component_index(self, v: str):
        """
        Get the index of a component.

        Parameters
        ----------
        v : str
            The name of the component.

        Returns
        -------
        int
            The index of the component.
        """

        return self._components.index(v)

    def component_name(self, i: int):
        """
        Get the name of a component.

        Parameters
        ----------
        i : int
            The index of the component.

        Returns
        -------
        str
            The name of the component.
        """
        return self._components[i]

    def positions(self, interior=False):
        """
        Get the positions of all cells in the domain.

        Returns
        -------
        list of float
            The positions of all cells in the domain.
        """
        return [cell.x() for cell in self.cells(interior)]

    def values(self, interior=False):
        """
        Get the values of all cells in the domain.

        Returns
        -------
        list of numpy.ndarray
            The values of all cells in the domain.
        """
        value_list = []
        for cell in self.cells(interior):
            value_list.append(cell.values())

        return value_list

    def listify_interior(self, split, split_loc):
        """
        Get the values of all interior cells in the domain in a list.

        Parameters
        ----------
        split : bool
            Whether or not to split the values in each cell into two lists.
        split_loc : int, optional
            The location to split the values in each cell, if `split` is `True`.

        Returns
        -------
        numpy.ndarray
            The values of all interior cells in the domain in a list.
        """
        nb_left, nb_right = self.nb(btype.LEFT), self.nb(btype.RIGHT)
        interior_values = self.values()[nb_left: -nb_right]

        if not split:
            return np.array(interior_values).flatten()
        else:
            if split_loc is None:
                raise SFXM("Split location must be specified in this case")

            num_points = len(interior_values)
            ret = []
            # First add all the outer-block values
            for i in range(num_points):
                ret.extend(interior_values[i][:split_loc])
            # Then add all the inner block values
            for i in range(num_points):
                ret.extend(interior_values[i][split_loc:])

            return np.array(ret)

    def update(self, dt, interior_residual_block):
        """
        Update the values of all cells in the domain.

        Parameters
        ----------
        dt : float
            The time step size.
        interior_residual_block : list of float
            The residuals for the interior cells in the domain.
        """

        for i, cell in enumerate(self.interior()):
            cell.update(dt, interior_residual_block[i])
